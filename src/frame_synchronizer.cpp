#include "frame_synchronizer.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <utility>

FrameSynchronizer::FrameSynchronizer(
    const std::array<WarpProducer *, 4> &producers)
    : producers_(producers), next_group_id_(0)
{
}

bool FrameSynchronizer::acquire_group(SynchronizedFrameGroup *output,
                                      int timeout_ms,
                                      uint64_t maximum_spread_ns)
{
    if (!output || timeout_ms < 0 || maximum_spread_ns == 0)
        return false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        bool fill_failed = false;
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i].valid())
                continue;
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                fill_failed = true;
                break;
            }
            const int remaining_ms = std::max(
                1,
                static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - now)
                        .count()));
            if (!producers_[i] ||
                !producers_[i]->acquire_latest(&pending_[i], remaining_ms)) {
                fill_failed = true;
                break;
            }
        }
        if (fill_failed) {
            ++stats_.acquire_timeouts;
            return false;
        }

        std::size_t oldest_index = 0;
        uint64_t minimum = std::numeric_limits<uint64_t>::max();
        uint64_t maximum = 0;
        std::set<int> camera_ids;
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            const uint64_t timestamp = pending_[i].timestamp_ns();
            if (timestamp < minimum) {
                minimum = timestamp;
                oldest_index = i;
            }
            maximum = std::max(maximum, timestamp);
            camera_ids.insert(pending_[i].camera_id());
        }
        if (camera_ids.size() != pending_.size()) {
            ++stats_.invalid_camera_sets;
            for (WarpedFrameRef &frame : pending_)
                frame.release();
            return false;
        }

        if (maximum - minimum <= maximum_spread_ns) {
            for (std::size_t i = 0; i < pending_.size(); ++i)
                output->frames[i] = std::move(pending_[i]);
            output->group_id = ++next_group_id_;
            output->min_timestamp_ns = minimum;
            output->max_timestamp_ns = maximum;
            ++stats_.groups;
            return true;
        }

        pending_[oldest_index].release();
        ++stats_.rejected_old_frames;
    }

    ++stats_.acquire_timeouts;
    return false;
}
