#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "camer_pip.h"
#include "gpu_warp_roi.h"

struct WarpConfig {
    std::string video_device;
    std::string camera_name;
    int camera_id = 0;
    int source_width = 1280;
    int source_height = 720;
    int source_stride = 1280;
    int roi_width = 0;
    int roi_height = 720;
    int roi_stride = 0;
    int panorama_x = 0;
    std::string map_x_path;
    std::string map_y_path;
    std::string valid_path;
};

struct WarpProducerStats {
    uint64_t captured = 0;
    uint64_t decode_drops = 0;
    uint64_t produced = 0;
    uint64_t overwritten_ready = 0;
    uint64_t no_writable_slot = 0;
    uint64_t consumer_acquired = 0;
    uint64_t consumer_skipped_ready = 0;
    uint64_t invalid_transitions = 0;
    uint64_t fatal_errors = 0;
    std::size_t free_slots = 0;
    std::size_t writing_slots = 0;
    std::size_t ready_slots = 0;
    std::size_t reading_slots = 0;
};

class WarpProducer;

class WarpedFrameRef {
public:
    WarpedFrameRef();
    ~WarpedFrameRef();
    WarpedFrameRef(WarpedFrameRef &&other) noexcept;
    WarpedFrameRef &operator=(WarpedFrameRef &&other) noexcept;

    WarpedFrameRef(const WarpedFrameRef &) = delete;
    WarpedFrameRef &operator=(const WarpedFrameRef &) = delete;

    bool valid() const { return owner_ != nullptr; }
    void release();

    int fd() const { return fd_; }
    int camera_id() const { return camera_id_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return stride_; }
    int panorama_x() const { return panorama_x_; }
    uint32_t sequence() const { return sequence_; }
    uint64_t timestamp_ns() const { return timestamp_ns_; }
    uint64_t publish_id() const { return publish_id_; }
    std::size_t slot() const { return slot_; }

private:
    friend class WarpProducer;
    WarpProducer *owner_;
    std::size_t slot_;
    uint64_t generation_;
    int fd_;
    int camera_id_;
    int width_;
    int height_;
    int stride_;
    int panorama_x_;
    uint32_t sequence_;
    uint64_t timestamp_ns_;
    uint64_t publish_id_;
};

class WarpProducer {
public:
    static constexpr std::size_t kSlotCount = 3;

    WarpProducer();
    ~WarpProducer();
    WarpProducer(const WarpProducer &) = delete;
    WarpProducer &operator=(const WarpProducer &) = delete;

    bool init(const WarpConfig &config);
    bool start();
    void stop();
    void join();
    void close();

    bool acquire_latest(WarpedFrameRef *output, int timeout_ms);
    WarpProducerStats stats() const;
    bool running() const { return running_.load(); }
    bool fatal() const { return fatal_.load(); }

private:
    enum class SlotState {
        Free,
        Writing,
        Ready,
        Reading,
    };

    struct Slot {
        SlotState state = SlotState::Free;
        uint64_t generation = 0;
        uint64_t publish_id = 0;
        uint32_t sequence = 0;
        uint64_t timestamp_ns = 0;
    };

    void worker_loop();
    bool claim_writable_slot(std::size_t *slot);
    void publish_slot(std::size_t slot, uint32_t sequence, uint64_t timestamp_ns);
    void abandon_slot(std::size_t slot);
    void release_slot(std::size_t slot, uint64_t generation);
    static uint64_t timeval_to_ns(const struct timeval &value);

    friend class WarpedFrameRef;

    WarpConfig config_;
    CameraPipe camera_;
    GpuWarpRoi warp_;
    std::array<Slot, kSlotCount> slots_;

    mutable std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> fatal_;
    bool initialized_;
    uint64_t next_publish_id_;
    WarpProducerStats counters_;
};
