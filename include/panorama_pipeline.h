#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using PanoramaReleaseCallback =
    void (*)(void *context, uint64_t frame_id) noexcept;

struct PanoramaFrame {
    int dma_fd = -1;
    void *data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    uint64_t sequence = 0;
    uint64_t timestamp_ns = 0;
    uint64_t camera_spread_ns = 0;
};

class PanoramaFrameRef {
public:
    PanoramaFrameRef();
    ~PanoramaFrameRef();
    PanoramaFrameRef(PanoramaFrameRef &&other) noexcept;
    PanoramaFrameRef &operator=(PanoramaFrameRef &&other) noexcept;

    PanoramaFrameRef(const PanoramaFrameRef &) = delete;
    PanoramaFrameRef &operator=(const PanoramaFrameRef &) = delete;

    bool valid() const { return release_callback_ != nullptr && dma_fd_ >= 0; }
    void release() noexcept;
    void detach() noexcept;

    int dma_fd() const { return dma_fd_; }
    void *data() const { return data_; }
    std::size_t bytes() const { return bytes_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return stride_; }
    uint64_t sequence() const { return sequence_; }
    uint64_t timestamp_ns() const { return timestamp_ns_; }
    uint64_t camera_spread_ns() const { return camera_spread_ns_; }
    PanoramaReleaseCallback release_callback() const {
        return release_callback_;
    }
    void *release_context() const { return release_context_; }

private:
    friend class PanoramaPipeline;
    int dma_fd_;
    void *data_;
    std::size_t bytes_;
    int width_;
    int height_;
    int stride_;
    uint64_t sequence_;
    uint64_t timestamp_ns_;
    uint64_t camera_spread_ns_;
    PanoramaReleaseCallback release_callback_;
    void *release_context_;
};

struct PanoramaPipelineStats {
    uint64_t frames = 0;
    uint64_t timeouts = 0;
    uint64_t errors = 0;
    uint64_t output_wait_timeouts = 0;
    uint64_t invalid_output_releases = 0;
    uint64_t outstanding_output_leases = 0;
    uint64_t sync_rejected_old_frames = 0;
    double rga_body_average_ms = 0.0;
    double gpu_seam_average_ms = 0.0;
    double bgr_convert_average_ms = 0.0;
};

class PanoramaPipeline {
public:
    PanoramaPipeline();
    ~PanoramaPipeline();
    PanoramaPipeline(const PanoramaPipeline &) = delete;
    PanoramaPipeline &operator=(const PanoramaPipeline &) = delete;

    bool init(const std::string &assets_directory);
    bool start();

    // Acquires a leased BGR DMA-BUF. The caller must either release() it or
    // transfer its callback/context to the asynchronous downstream consumer.
    bool acquire(PanoramaFrameRef *frame, int timeout_ms = 1000);

    // The returned DMA-BUF and virtual address remain owned by this object.
    // Their image contents are valid until the next successful read(). This
    // legacy wrapper internally holds one PanoramaFrameRef lease.
    bool read(PanoramaFrame *frame, int timeout_ms = 1000);

    void stop();
    void close();
    bool fatal() const;
    PanoramaPipelineStats stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
