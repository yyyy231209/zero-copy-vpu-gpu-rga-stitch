#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "warp_producer.h"

struct SynchronizedFrameGroup {
    std::array<WarpedFrameRef, 4> frames;
    uint64_t group_id = 0;
    uint64_t min_timestamp_ns = 0;
    uint64_t max_timestamp_ns = 0;

    uint64_t spread_ns() const { return max_timestamp_ns - min_timestamp_ns; }
};

struct FrameSynchronizerStats {
    uint64_t groups = 0;
    uint64_t acquire_timeouts = 0;
    uint64_t rejected_old_frames = 0;
    uint64_t invalid_camera_sets = 0;
};

class FrameSynchronizer {
public:
    explicit FrameSynchronizer(const std::array<WarpProducer *, 4> &producers);

    bool acquire_group(SynchronizedFrameGroup *output,
                       int timeout_ms,
                       uint64_t maximum_spread_ns);
    FrameSynchronizerStats stats() const { return stats_; }

private:
    std::array<WarpProducer *, 4> producers_;
    std::array<WarpedFrameRef, 4> pending_;
    uint64_t next_group_id_;
    FrameSynchronizerStats stats_;
};
