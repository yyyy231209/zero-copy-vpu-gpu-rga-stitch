#include "panorama_pipeline.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "camera_resolver.h"
#include "frame_synchronizer.h"
#include "panorama_composer.h"
#include "warp_producer.h"

namespace {

struct CameraSpec {
    int logical;
    int hub_port;
    int roi_width;
    int panorama_x;
    const char *asset_prefix;
};

constexpr CameraSpec kCameraSpecs[4] = {
    {2, 4, 688, 84, "cam2"},
    {1, 3, 436, 738, "cam1"},
    {4, 2, 588, 1140, "cam4"},
    {3, 5, 640, 1694, "cam3"},
};

}  // namespace

PanoramaFrameRef::PanoramaFrameRef()
    : dma_fd_(-1), data_(nullptr), bytes_(0), width_(0), height_(0),
      stride_(0), sequence_(0), timestamp_ns_(0), camera_spread_ns_(0),
      release_callback_(nullptr), release_context_(nullptr)
{
}

PanoramaFrameRef::~PanoramaFrameRef() { release(); }

PanoramaFrameRef::PanoramaFrameRef(PanoramaFrameRef &&other) noexcept
    : PanoramaFrameRef()
{
    *this = std::move(other);
}

PanoramaFrameRef &PanoramaFrameRef::operator=(PanoramaFrameRef &&other) noexcept
{
    if (this == &other)
        return *this;
    release();
    dma_fd_ = other.dma_fd_;
    data_ = other.data_;
    bytes_ = other.bytes_;
    width_ = other.width_;
    height_ = other.height_;
    stride_ = other.stride_;
    sequence_ = other.sequence_;
    timestamp_ns_ = other.timestamp_ns_;
    camera_spread_ns_ = other.camera_spread_ns_;
    release_callback_ = other.release_callback_;
    release_context_ = other.release_context_;
    other.detach();
    return *this;
}

void PanoramaFrameRef::release() noexcept
{
    const PanoramaReleaseCallback callback = release_callback_;
    void *context = release_context_;
    const uint64_t sequence = sequence_;
    detach();
    if (callback)
        callback(context, sequence);
}

void PanoramaFrameRef::detach() noexcept
{
    dma_fd_ = -1;
    data_ = nullptr;
    bytes_ = 0;
    width_ = 0;
    height_ = 0;
    stride_ = 0;
    sequence_ = 0;
    timestamp_ns_ = 0;
    camera_spread_ns_ = 0;
    release_callback_ = nullptr;
    release_context_ = nullptr;
}

class PanoramaPipeline::Impl {
public:
    enum class OutputState {
        Free,
        Writing,
        Leased,
    };
    struct ReleaseContext {
        Impl *owner = nullptr;
        std::size_t slot = 0;
        uint64_t generation = 0;
    };
    struct OutputLease {
        OutputState state = OutputState::Free;
        uint64_t generation = 0;
        ReleaseContext release;
    };

    static void release_callback(void *context, uint64_t) noexcept
    {
        auto *release = static_cast<ReleaseContext *>(context);
        if (release && release->owner) {
            release->owner->release_output(
                release->slot, release->generation);
        }
    }

    void reset_outputs()
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        closing = false;
        outstanding_outputs = 0;
        invalid_output_releases = 0;
        for (std::size_t i = 0; i < output_leases.size(); ++i) {
            output_leases[i].state = OutputState::Free;
            output_leases[i].generation = 0;
            output_leases[i].release = {this, i, 0};
        }
    }

    bool claim_output(std::size_t *slot, uint64_t *generation,
                      int timeout_ms)
    {
        if (!slot || !generation || timeout_ms < 0)
            return false;
        std::unique_lock<std::mutex> lock(output_mutex);
        const auto available = [this] {
            if (closing)
                return true;
            for (const auto &entry : output_leases) {
                if (entry.state == OutputState::Free)
                    return true;
            }
            return false;
        };
        if (!output_cv.wait_for(
                lock, std::chrono::milliseconds(timeout_ms), available) ||
            closing)
            return false;
        for (std::size_t i = 0; i < output_leases.size(); ++i) {
            auto &entry = output_leases[i];
            if (entry.state != OutputState::Free)
                continue;
            entry.state = OutputState::Writing;
            ++entry.generation;
            entry.release = {this, i, entry.generation};
            ++outstanding_outputs;
            *slot = i;
            *generation = entry.generation;
            return true;
        }
        return false;
    }

    bool publish_output(std::size_t slot, uint64_t generation)
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (slot >= output_leases.size() ||
            output_leases[slot].state != OutputState::Writing ||
            output_leases[slot].generation != generation) {
            ++invalid_output_releases;
            return false;
        }
        output_leases[slot].state = OutputState::Leased;
        return true;
    }

    void cancel_output(std::size_t slot, uint64_t generation) noexcept
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (slot >= output_leases.size() ||
            output_leases[slot].state != OutputState::Writing ||
            output_leases[slot].generation != generation) {
            ++invalid_output_releases;
            return;
        }
        output_leases[slot].state = OutputState::Free;
        if (outstanding_outputs > 0)
            --outstanding_outputs;
        output_cv.notify_all();
    }

    void release_output(std::size_t slot, uint64_t generation) noexcept
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (slot >= output_leases.size() ||
            output_leases[slot].state != OutputState::Leased ||
            output_leases[slot].generation != generation) {
            ++invalid_output_releases;
            return;
        }
        output_leases[slot].state = OutputState::Free;
        if (outstanding_outputs > 0)
            --outstanding_outputs;
        output_cv.notify_all();
    }

    void close_outputs()
    {
        std::unique_lock<std::mutex> lock(output_mutex);
        closing = true;
        output_cv.notify_all();
        output_cv.wait(lock, [this] { return outstanding_outputs == 0; });
        for (auto &entry : output_leases)
            entry.release.owner = nullptr;
    }

    std::array<std::unique_ptr<WarpProducer>, 4> producers;
    std::array<WarpProducer *, 4> producer_ptrs{};
    std::unique_ptr<FrameSynchronizer> synchronizer;
    PanoramaComposer composer;
    std::array<OutputLease, PanoramaComposer::kBgrSlots> output_leases;
    std::mutex output_mutex;
    std::condition_variable output_cv;
    PanoramaFrameRef legacy_frame;
    std::string assets_directory;
    uint64_t frames = 0;
    uint64_t timeouts = 0;
    uint64_t consecutive_timeouts = 0;
    uint64_t errors = 0;
    uint64_t output_wait_timeouts = 0;
    uint64_t invalid_output_releases = 0;
    uint64_t outstanding_outputs = 0;
    bool closing = false;
    bool initialized = false;
    bool running = false;
    bool fatal_error = false;
};

PanoramaPipeline::PanoramaPipeline() : impl_(std::make_unique<Impl>()) {}

PanoramaPipeline::~PanoramaPipeline() { close(); }

bool PanoramaPipeline::init(const std::string &assets_directory)
{
    if (impl_->initialized)
        return false;

    impl_->assets_directory = assets_directory;
    impl_->reset_outputs();
    for (std::size_t i = 0; i < impl_->producers.size(); ++i) {
        const auto &spec = kCameraSpecs[i];
        std::string device;
        std::string diagnostic;
        if (!resolve_usb_camera_by_hub_port(
                spec.hub_port, 1280, 720, &device, &diagnostic)) {
            std::fprintf(stderr, "resolve hub .%d failed: %s\n",
                         spec.hub_port, diagnostic.c_str());
            close();
            return false;
        }

        WarpConfig config;
        config.video_device = device;
        config.camera_id = spec.logical;
        config.camera_name = "live-cam" + std::to_string(spec.logical);
        config.roi_width = spec.roi_width;
        config.roi_stride = (config.roi_width + 63) / 64 * 64;
        config.roi_height = PanoramaComposer::kHeight;
        config.panorama_x = spec.panorama_x;
        const std::string prefix =
            assets_directory + "/" + spec.asset_prefix;
        config.map_x_path = prefix + "_map_x.f32";
        config.map_y_path = prefix + "_map_y.f32";
        config.valid_path = prefix + "_valid.u8";

        impl_->producers[i] = std::make_unique<WarpProducer>();
        if (!impl_->producers[i]->init(config)) {
            close();
            return false;
        }
        impl_->producer_ptrs[i] = impl_->producers[i].get();
        std::printf("cam%d hub .%d -> %s\n",
                    spec.logical, spec.hub_port, device.c_str());
    }

    if (!impl_->composer.init(assets_directory)) {
        close();
        return false;
    }
    impl_->synchronizer =
        std::make_unique<FrameSynchronizer>(impl_->producer_ptrs);
    impl_->initialized = true;
    return true;
}

bool PanoramaPipeline::start()
{
    if (!impl_->initialized || impl_->running)
        return false;
    for (auto &producer : impl_->producers) {
        if (!producer->start()) {
            impl_->fatal_error = true;
            stop();
            return false;
        }
    }
    impl_->running = true;
    return true;
}

bool PanoramaPipeline::read(PanoramaFrame *frame, int timeout_ms)
{
    if (!frame)
        return false;

    impl_->legacy_frame.release();
    PanoramaFrameRef leased;
    if (!acquire(&leased, timeout_ms))
        return false;

    frame->dma_fd = leased.dma_fd();
    frame->data = leased.data();
    frame->width = leased.width();
    frame->height = leased.height();
    frame->stride = leased.stride();
    frame->sequence = leased.sequence();
    frame->timestamp_ns = leased.timestamp_ns();
    frame->camera_spread_ns = leased.camera_spread_ns();
    impl_->legacy_frame = std::move(leased);
    return true;
}

bool PanoramaPipeline::acquire(PanoramaFrameRef *frame, int timeout_ms)
{
    if (!frame || frame->valid() || !impl_->running || impl_->fatal_error)
        return false;

    std::size_t bgr_slot = 0;
    uint64_t generation = 0;
    if (!impl_->claim_output(&bgr_slot, &generation, timeout_ms)) {
        ++impl_->output_wait_timeouts;
        return false;
    }

    SynchronizedFrameGroup group;
    if (!impl_->synchronizer->acquire_group(
            &group, timeout_ms, UINT64_C(40000000))) {
        impl_->cancel_output(bgr_slot, generation);
        ++impl_->timeouts;
        ++impl_->consecutive_timeouts;
        for (const auto &producer : impl_->producers)
            impl_->fatal_error |= producer->fatal();
        if (impl_->consecutive_timeouts > 20)
            impl_->fatal_error = true;
        return false;
    }
    impl_->consecutive_timeouts = 0;

    const std::size_t slot =
        impl_->frames % PanoramaComposer::kOutputSlots;
    if (!impl_->composer.compose(group.frames, slot) ||
        !impl_->composer.convert_bgr(slot, bgr_slot)) {
        impl_->cancel_output(bgr_slot, generation);
        ++impl_->errors;
        impl_->fatal_error = true;
        return false;
    }

    ++impl_->frames;
    if (!impl_->publish_output(bgr_slot, generation)) {
        impl_->cancel_output(bgr_slot, generation);
        ++impl_->errors;
        impl_->fatal_error = true;
        return false;
    }
    frame->dma_fd_ = impl_->composer.bgr_fd(bgr_slot);
    frame->data_ = impl_->composer.bgr_va(bgr_slot);
    frame->bytes_ = impl_->composer.bgr_size();
    frame->width_ = PanoramaComposer::kBgrWidth;
    frame->height_ = PanoramaComposer::kBgrHeight;
    frame->stride_ = PanoramaComposer::kBgrStride;
    frame->sequence_ = impl_->frames;
    frame->timestamp_ns_ = group.max_timestamp_ns;
    frame->camera_spread_ns_ = group.spread_ns();
    frame->release_callback_ = &Impl::release_callback;
    frame->release_context_ =
        &impl_->output_leases[bgr_slot].release;
    return true;
}

void PanoramaPipeline::stop()
{
    for (auto &producer : impl_->producers) {
        if (producer)
            producer->stop();
    }
    for (auto &producer : impl_->producers) {
        if (producer)
            producer->join();
    }
    impl_->running = false;
}

void PanoramaPipeline::close()
{
    stop();
    impl_->legacy_frame.release();
    impl_->close_outputs();
    impl_->synchronizer.reset();
    impl_->composer.close();
    for (auto &producer : impl_->producers) {
        if (producer)
            producer->close();
        producer.reset();
    }
    impl_->producer_ptrs.fill(nullptr);
    impl_->initialized = false;
}

bool PanoramaPipeline::fatal() const { return impl_->fatal_error; }

PanoramaPipelineStats PanoramaPipeline::stats() const
{
    PanoramaPipelineStats result;
    result.frames = impl_->frames;
    result.timeouts = impl_->timeouts;
    result.errors = impl_->errors;
    result.output_wait_timeouts = impl_->output_wait_timeouts;
    {
        std::lock_guard<std::mutex> lock(impl_->output_mutex);
        result.invalid_output_releases = impl_->invalid_output_releases;
        result.outstanding_output_leases = impl_->outstanding_outputs;
    }
    if (impl_->synchronizer) {
        result.sync_rejected_old_frames =
            impl_->synchronizer->stats().rejected_old_frames;
    }
    const auto compose = impl_->composer.stats();
    if (compose.frames > 0) {
        result.rga_body_average_ms =
            compose.rga_body_total_ms / compose.frames;
        result.gpu_seam_average_ms =
            compose.gpu_seam_total_ms / compose.frames;
        result.bgr_convert_average_ms =
            compose.bgr_convert_total_ms / compose.frames;
    }
    return result;
}
