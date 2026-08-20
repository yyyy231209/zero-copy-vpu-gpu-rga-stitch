#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstddef>
#include <string>
#include <vector>

struct GpuWarpRoiConfig {
    int source_width = 0;
    int source_height = 0;
    int source_y_stride = 0;
    int source_uv_stride = 0;
    int destination_width = 0;
    int destination_height = 0;
    int destination_y_stride = 0;
    int destination_uv_stride = 0;
    std::string map_x_path;
    std::string map_y_path;
    std::string valid_path;
    int output_buffer_count = 1;
};

class GpuWarpRoi {
public:
    GpuWarpRoi();
    ~GpuWarpRoi();

    GpuWarpRoi(const GpuWarpRoi &) = delete;
    GpuWarpRoi &operator=(const GpuWarpRoi &) = delete;

    bool init(int source_dma_fd, const GpuWarpRoiConfig &config);
    bool execute(std::size_t output_index = 0);
    void close();

    int output_fd(std::size_t output_index = 0) const;
    void *output_va(std::size_t output_index = 0) const;
    std::size_t output_size() const { return destination_size_; }
    std::size_t output_count() const { return destination_fds_.size(); }
    const GpuWarpRoiConfig &config() const { return config_; }

private:
    bool init_opencl();
    bool load_maps();
    bool build_kernels();
    bool import_buffers(int source_dma_fd);
    bool set_kernel_arguments();
    bool validate_config() const;

    GpuWarpRoiConfig config_;

    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    cl_program program_;
    cl_kernel y_kernel_;
    cl_kernel uv_kernel_;
    cl_mem source_cl_;
    cl_mem map_x_cl_;
    cl_mem map_y_cl_;
    cl_mem valid_cl_;

    std::vector<int> destination_fds_;
    std::vector<void *> destination_vas_;
    std::vector<cl_mem> destination_cl_;
    std::size_t source_size_;
    std::size_t destination_size_;
    bool initialized_;
};
