#include "warp_producer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>

#include <rockchip/mpp_buffer.h>

WarpedFrameRef::WarpedFrameRef()
    : owner_(nullptr),
      slot_(0),
      generation_(0),
      fd_(-1),
      camera_id_(0),
      width_(0),
      height_(0),
      stride_(0),
      panorama_x_(0),
      sequence_(0),
      timestamp_ns_(0),
      publish_id_(0)
{
}

WarpedFrameRef::~WarpedFrameRef()
{
    release();
}

WarpedFrameRef::WarpedFrameRef(WarpedFrameRef &&other) noexcept
    : WarpedFrameRef()
{
    *this = std::move(other);
}

WarpedFrameRef &WarpedFrameRef::operator=(WarpedFrameRef &&other) noexcept
{
    if (this == &other)
        return *this;
    release();
    owner_ = other.owner_;
    slot_ = other.slot_;
    generation_ = other.generation_;
    fd_ = other.fd_;
    camera_id_ = other.camera_id_;
    width_ = other.width_;
    height_ = other.height_;
    stride_ = other.stride_;
    panorama_x_ = other.panorama_x_;
    sequence_ = other.sequence_;
    timestamp_ns_ = other.timestamp_ns_;
    publish_id_ = other.publish_id_;
    other.owner_ = nullptr;
    other.fd_ = -1;
    return *this;
}

void WarpedFrameRef::release()
{
    if (owner_)
        owner_->release_slot(slot_, generation_);
    owner_ = nullptr;
    fd_ = -1;
}

WarpProducer::WarpProducer()
    : camera_{},
      running_(false),
      stop_requested_(false),
      fatal_(false),
      initialized_(false),
      next_publish_id_(0)
{
    camera_.fd = -1;
    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i)
        camera_.buffers[i].dma_fd = -1;
}

WarpProducer::~WarpProducer()
{
    close();
}

bool WarpProducer::init(const WarpConfig &config)
{
    if (initialized_) {
        std::fprintf(stderr, "WarpProducer: init called twice\n");
        return false;
    }
    if (config.video_device.empty() || config.camera_name.empty() ||
        config.roi_width <= 0 || config.roi_height <= 0 ||
        config.roi_stride < config.roi_width ||
        (config.roi_width % 2) != 0 || (config.roi_height % 2) != 0 ||
        (config.roi_stride % 2) != 0) {
        std::fprintf(stderr, "WarpProducer: invalid configuration\n");
        return false;
    }

    config_ = config;
    stop_requested_.store(false);
    fatal_.store(false);
    counters_ = {};
    next_publish_id_ = 0;
    for (Slot &slot : slots_)
        slot = {};

    if (cp_open(&camera_, config_.video_device.c_str()) < 0) {
        std::fprintf(stderr, "[%s] camera initialization failed\n",
                     config_.camera_name.c_str());
        close();
        return false;
    }
    if (camera_.w != config_.source_width ||
        camera_.h != config_.source_height ||
        !camera_.dec.out_buf) {
        std::fprintf(stderr,
                     "[%s] camera/MPP geometry mismatch: got %dx%d\n",
                     config_.camera_name.c_str(), camera_.w, camera_.h);
        close();
        return false;
    }
    const int source_fd = mpp_buffer_get_fd(camera_.dec.out_buf);
    if (source_fd < 0) {
        std::fprintf(stderr, "[%s] invalid MPP output DMA-BUF fd\n",
                     config_.camera_name.c_str());
        close();
        return false;
    }

    GpuWarpRoiConfig gpu;
    gpu.source_width = config_.source_width;
    gpu.source_height = config_.source_height;
    gpu.source_y_stride = config_.source_stride;
    gpu.source_uv_stride = config_.source_stride;
    gpu.destination_width = config_.roi_width;
    gpu.destination_height = config_.roi_height;
    gpu.destination_y_stride = config_.roi_stride;
    gpu.destination_uv_stride = config_.roi_stride;
    gpu.map_x_path = config_.map_x_path;
    gpu.map_y_path = config_.map_y_path;
    gpu.valid_path = config_.valid_path;
    gpu.output_buffer_count = static_cast<int>(kSlotCount);
    if (!warp_.init(source_fd, gpu)) {
        std::fprintf(stderr, "[%s] GPU ROI initialization failed\n",
                     config_.camera_name.c_str());
        close();
        return false;
    }

    initialized_ = true;
    return true;
}

bool WarpProducer::start()
{
    if (!initialized_ || worker_.joinable()) {
        std::fprintf(stderr, "WarpProducer: start requires initialized idle producer\n");
        return false;
    }
    stop_requested_.store(false);
    fatal_.store(false);
    running_.store(true);
    try {
        worker_ = std::thread(&WarpProducer::worker_loop, this);
    } catch (...) {
        running_.store(false);
        std::fprintf(stderr, "WarpProducer: worker thread creation failed\n");
        return false;
    }
    return true;
}

void WarpProducer::stop()
{
    stop_requested_.store(true);
    ready_cv_.notify_all();
}

void WarpProducer::join()
{
    if (worker_.joinable())
        worker_.join();
}

uint64_t WarpProducer::timeval_to_ns(const struct timeval &value)
{
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
           static_cast<uint64_t>(value.tv_usec) * UINT64_C(1000);
}

bool WarpProducer::claim_writable_slot(std::size_t *slot_index)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == SlotState::Free) {
            slots_[i].state = SlotState::Writing;
            ++slots_[i].generation;
            *slot_index = i;
            return true;
        }
    }

    std::size_t oldest = slots_.size();
    uint64_t oldest_publish = std::numeric_limits<uint64_t>::max();
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == SlotState::Ready &&
            slots_[i].publish_id < oldest_publish) {
            oldest = i;
            oldest_publish = slots_[i].publish_id;
        }
    }
    if (oldest != slots_.size()) {
        ++counters_.overwritten_ready;
        slots_[oldest].state = SlotState::Writing;
        ++slots_[oldest].generation;
        *slot_index = oldest;
        return true;
    }

    ++counters_.no_writable_slot;
    return false;
}

void WarpProducer::publish_slot(std::size_t slot_index,
                                uint32_t sequence,
                                uint64_t timestamp_ns)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Slot &slot = slots_[slot_index];
        if (slot.state != SlotState::Writing) {
            ++counters_.invalid_transitions;
            return;
        }
        slot.sequence = sequence;
        slot.timestamp_ns = timestamp_ns;
        slot.publish_id = ++next_publish_id_;
        slot.state = SlotState::Ready;
        ++counters_.produced;
    }
    ready_cv_.notify_one();
}

void WarpProducer::abandon_slot(std::size_t slot_index)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_[slot_index].state != SlotState::Writing)
        ++counters_.invalid_transitions;
    else
        slots_[slot_index].state = SlotState::Free;
}

void WarpProducer::worker_loop()
{
    while (!stop_requested_.load()) {
        const int result = cp_next_status(&camera_);
        if (result == CP_NEXT_TIMEOUT)
            continue;
        if (result == CP_NEXT_DROP) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.decode_drops;
            continue;
        }
        if (result != CP_NEXT_OK) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.fatal_errors;
            fatal_.store(true);
            break;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.captured;
        }

        std::size_t slot_index = 0;
        if (!claim_writable_slot(&slot_index))
            continue;
        if (!warp_.execute(slot_index)) {
            abandon_slot(slot_index);
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.fatal_errors;
            fatal_.store(true);
            break;
        }
        if (!camera_.have_frame_metadata) {
            abandon_slot(slot_index);
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.fatal_errors;
            fatal_.store(true);
            break;
        }
        publish_slot(slot_index, camera_.last_sequence,
                     timeval_to_ns(camera_.last_timestamp));
    }
    running_.store(false);
    ready_cv_.notify_all();
}

bool WarpProducer::acquire_latest(WarpedFrameRef *output, int timeout_ms)
{
    if (!output || timeout_ms < 0)
        return false;
    output->release();

    std::unique_lock<std::mutex> lock(mutex_);
    const auto has_ready_or_stopped = [this]() {
        if (!running_.load() || fatal_.load())
            return true;
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot &slot) {
            return slot.state == SlotState::Ready;
        });
    };
    if (!ready_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            has_ready_or_stopped))
        return false;

    std::size_t newest = slots_.size();
    uint64_t newest_publish = 0;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == SlotState::Ready &&
            slots_[i].publish_id > newest_publish) {
            newest = i;
            newest_publish = slots_[i].publish_id;
        }
    }
    if (newest == slots_.size())
        return false;

    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (i != newest && slots_[i].state == SlotState::Ready) {
            slots_[i].state = SlotState::Free;
            ++counters_.consumer_skipped_ready;
        }
    }
    Slot &slot = slots_[newest];
    slot.state = SlotState::Reading;
    ++counters_.consumer_acquired;

    output->owner_ = this;
    output->slot_ = newest;
    output->generation_ = slot.generation;
    output->fd_ = warp_.output_fd(newest);
    output->camera_id_ = config_.camera_id;
    output->width_ = config_.roi_width;
    output->height_ = config_.roi_height;
    output->stride_ = config_.roi_stride;
    output->panorama_x_ = config_.panorama_x;
    output->sequence_ = slot.sequence;
    output->timestamp_ns_ = slot.timestamp_ns;
    output->publish_id_ = slot.publish_id;
    return true;
}

void WarpProducer::release_slot(std::size_t slot_index, uint64_t generation)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_index >= slots_.size() ||
        slots_[slot_index].state != SlotState::Reading ||
        slots_[slot_index].generation != generation) {
        ++counters_.invalid_transitions;
        return;
    }
    slots_[slot_index].state = SlotState::Free;
}

WarpProducerStats WarpProducer::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    WarpProducerStats result = counters_;
    for (const Slot &slot : slots_) {
        switch (slot.state) {
        case SlotState::Free:
            ++result.free_slots;
            break;
        case SlotState::Writing:
            ++result.writing_slots;
            break;
        case SlotState::Ready:
            ++result.ready_slots;
            break;
        case SlotState::Reading:
            ++result.reading_slots;
            break;
        }
    }
    return result;
}

void WarpProducer::close()
{
    stop();
    join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (Slot &slot : slots_) {
            if (slot.state == SlotState::Reading) {
                ++counters_.invalid_transitions;
                std::fprintf(stderr,
                             "WarpProducer: close called with outstanding FrameRef\n");
            }
            slot.state = SlotState::Free;
        }
    }
    warp_.close();
    cp_close(&camera_);
    initialized_ = false;
    running_.store(false);
}
