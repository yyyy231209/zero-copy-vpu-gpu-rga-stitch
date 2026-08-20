#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "warp_producer.h"

struct PanoramaComposeStats {
    uint64_t frames = 0;
    double rga_body_total_ms = 0.0;
    double gpu_seam_total_ms = 0.0;
    double bgr_convert_total_ms = 0.0;
};

class PanoramaComposer {
public:
    static constexpr std::size_t kOutputSlots = 3;
    static constexpr std::size_t kBgrSlots = 6;
    static constexpr int kVisibleWidth = 2389;
    static constexpr int kWidth = 2390;
    static constexpr int kHeight = 720;
    static constexpr int kStride = 2432;
    static constexpr int kBgrCropX = 84;
    static constexpr int kBgrCropY = 198;
    static constexpr int kBgrWidth = 2248;
    static constexpr int kBgrHeight = 330;
    static constexpr int kBgrStride = 2256;

    PanoramaComposer();
    ~PanoramaComposer();
    PanoramaComposer(const PanoramaComposer &) = delete;
    PanoramaComposer &operator=(const PanoramaComposer &) = delete;

    bool init(const std::string &assets_directory);
    bool compose(const std::array<WarpedFrameRef, 4> &frames,
                 std::size_t output_slot);
    bool convert_bgr(std::size_t output_slot, std::size_t bgr_slot);
    void close();

    int output_fd(std::size_t slot) const;
    void *output_va(std::size_t slot) const;
    std::size_t output_size() const { return output_size_; }
    void *bgr_va(std::size_t slot) const;
    int bgr_fd(std::size_t slot) const;
    std::size_t bgr_size() const { return bgr_size_; }
    PanoramaComposeStats stats() const { return stats_; }

private:
    struct DmaOutput {
        int fd = -1;
        void *va = nullptr;
        cl_mem memory = nullptr;
    };
    struct BgrOutput {
        int fd = -1;
        void *va = nullptr;
    };
    struct SeamResources {
        cl_mem left_y = nullptr;
        cl_mem right_y = nullptr;
        cl_mem coverage_y = nullptr;
        cl_mem left_uv = nullptr;
        cl_mem right_uv = nullptr;
        cl_mem coverage_uv = nullptr;
    };

    bool init_opencl();
    bool load_weights(const std::string &assets_directory);
    cl_mem import_input(int fd, std::size_t size);
    bool run_rga_body(const std::array<WarpedFrameRef, 4> &frames,
                      std::size_t slot);
    bool run_gpu_seams(const std::array<WarpedFrameRef, 4> &frames,
                       std::size_t slot);

    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    cl_program program_;
    cl_kernel y_kernel_;
    cl_kernel uv_kernel_;
    std::array<DmaOutput, kOutputSlots> outputs_;
    std::array<BgrOutput, kBgrSlots> bgr_outputs_;
    std::array<SeamResources, 3> seams_;
    std::unordered_map<int, cl_mem> input_cache_;
    std::size_t output_size_;
    std::size_t bgr_size_;
    bool initialized_;
    PanoramaComposeStats stats_;
};
