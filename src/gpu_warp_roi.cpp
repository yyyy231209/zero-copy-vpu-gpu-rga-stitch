#include "gpu_warp_roi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dma_alloc.h"

namespace {

const char *kKernelSource = R"CLC(
__kernel void warp_y(
    __global const uchar *src,
    __global uchar *dst,
    __global const float *map_x,
    __global const float *map_y,
    __global const uchar *valid,
    int src_w,
    int src_h,
    int src_y_stride,
    int dst_w,
    int dst_h,
    int dst_y_stride)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= dst_w || y >= dst_h)
        return;

    const int index = y * dst_w + x;
    if (!valid[index])
        return;

    const float u = map_x[index];
    const float v = map_y[index];
    if (u < 0.0f || u >= (float)(src_w - 1) ||
        v < 0.0f || v >= (float)(src_h - 1))
        return;

    const int iu = (int)u;
    const int iv = (int)v;
    const float fu = u - (float)iu;
    const float fv = v - (float)iv;
    const float w00 = (1.0f - fu) * (1.0f - fv);
    const float w01 = fu * (1.0f - fv);
    const float w10 = (1.0f - fu) * fv;
    const float w11 = fu * fv;

    const float p00 = (float)src[iv * src_y_stride + iu];
    const float p01 = (float)src[iv * src_y_stride + iu + 1];
    const float p10 = (float)src[(iv + 1) * src_y_stride + iu];
    const float p11 = (float)src[(iv + 1) * src_y_stride + iu + 1];
    dst[y * dst_y_stride + x] =
        convert_uchar_sat_rte(w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11);
}

__kernel void warp_uv(
    __global const uchar *src,
    __global uchar *dst,
    __global const float *map_x,
    __global const float *map_y,
    __global const uchar *valid,
    int src_w,
    int src_h,
    int src_y_stride,
    int src_uv_stride,
    int dst_w,
    int dst_h,
    int dst_y_stride,
    int dst_uv_stride)
{
    const int cx = get_global_id(0);
    const int cy = get_global_id(1);
    const int x = cx * 2;
    const int y = cy * 2;
    if (x >= dst_w || y >= dst_h)
        return;

    const int index = y * dst_w + x;
    if (!valid[index])
        return;

    const float u = map_x[index] * 0.5f;
    const float v = map_y[index] * 0.5f;
    const int src_cw = src_w / 2;
    const int src_ch = src_h / 2;
    if (u < 0.0f || u >= (float)(src_cw - 1) ||
        v < 0.0f || v >= (float)(src_ch - 1))
        return;

    const int iu = (int)u;
    const int iv = (int)v;
    const float fu = u - (float)iu;
    const float fv = v - (float)iv;
    const float w00 = (1.0f - fu) * (1.0f - fv);
    const float w01 = fu * (1.0f - fv);
    const float w10 = (1.0f - fu) * fv;
    const float w11 = fu * fv;
    const int src_uv_base = src_y_stride * src_h;
    const int dst_uv_base = dst_y_stride * dst_h;

    const int p00 = src_uv_base + iv * src_uv_stride + iu * 2;
    const int p01 = p00 + 2;
    const int p10 = p00 + src_uv_stride;
    const int p11 = p10 + 2;
    const int out = dst_uv_base + cy * dst_uv_stride + x;

    dst[out] = convert_uchar_sat_rte(
        w00 * (float)src[p00] + w01 * (float)src[p01] +
        w10 * (float)src[p10] + w11 * (float)src[p11]);
    dst[out + 1] = convert_uchar_sat_rte(
        w00 * (float)src[p00 + 1] + w01 * (float)src[p01 + 1] +
        w10 * (float)src[p10 + 1] + w11 * (float)src[p11 + 1]);
}
)CLC";

template <typename T>
bool read_exact(const std::string &path, std::size_t count, std::vector<T> *output)
{
    output->resize(count);
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "GpuWarpRoi: cannot open %s\n", path.c_str());
        return false;
    }
    const std::size_t actual = std::fread(output->data(), sizeof(T), count, file);
    const int trailing = std::fgetc(file);
    std::fclose(file);
    if (actual != count || trailing != EOF) {
        std::fprintf(stderr,
                     "GpuWarpRoi: invalid size for %s (read %zu elements, expected %zu)\n",
                     path.c_str(), actual, count);
        return false;
    }
    return true;
}

}  // namespace

GpuWarpRoi::GpuWarpRoi()
    : context_(nullptr),
      queue_(nullptr),
      device_(nullptr),
      program_(nullptr),
      y_kernel_(nullptr),
      uv_kernel_(nullptr),
      source_cl_(nullptr),
      map_x_cl_(nullptr),
      map_y_cl_(nullptr),
      valid_cl_(nullptr),
      source_size_(0),
      destination_size_(0),
      initialized_(false)
{
}

GpuWarpRoi::~GpuWarpRoi()
{
    close();
}

bool GpuWarpRoi::validate_config() const
{
    const bool dimensions_ok =
        config_.source_width > 1 && config_.source_height > 1 &&
        config_.destination_width > 0 && config_.destination_height > 0 &&
        (config_.source_width % 2) == 0 && (config_.source_height % 2) == 0 &&
        (config_.destination_width % 2) == 0 &&
        (config_.destination_height % 2) == 0;
    const bool strides_ok =
        config_.source_y_stride >= config_.source_width &&
        config_.source_uv_stride >= config_.source_width &&
        config_.destination_y_stride >= config_.destination_width &&
        config_.destination_uv_stride >= config_.destination_width &&
        (config_.source_y_stride % 2) == 0 &&
        (config_.source_uv_stride % 2) == 0 &&
        (config_.destination_y_stride % 2) == 0 &&
        (config_.destination_uv_stride % 2) == 0;
    const bool pool_ok =
        config_.output_buffer_count >= 1 && config_.output_buffer_count <= 16;
    const bool paths_ok = !config_.map_x_path.empty() &&
                          !config_.map_y_path.empty() &&
                          !config_.valid_path.empty();
    if (!dimensions_ok || !strides_ok || !paths_ok || !pool_ok) {
        std::fprintf(stderr, "GpuWarpRoi: invalid dimensions, strides, or map paths\n");
        return false;
    }
    return true;
}

bool GpuWarpRoi::init(int source_dma_fd, const GpuWarpRoiConfig &config)
{
    if (initialized_) {
        std::fprintf(stderr, "GpuWarpRoi: init called twice\n");
        return false;
    }
    if (source_dma_fd < 0) {
        std::fprintf(stderr, "GpuWarpRoi: invalid source DMA-BUF fd\n");
        return false;
    }

    config_ = config;
    if (!validate_config())
        return false;

    source_size_ =
        static_cast<std::size_t>(config_.source_y_stride) * config_.source_height +
        static_cast<std::size_t>(config_.source_uv_stride) *
            (config_.source_height / 2);
    destination_size_ =
        static_cast<std::size_t>(config_.destination_y_stride) *
            config_.destination_height +
        static_cast<std::size_t>(config_.destination_uv_stride) *
            (config_.destination_height / 2);

    if (!init_opencl() || !load_maps() || !build_kernels() ||
        !import_buffers(source_dma_fd) || !set_kernel_arguments()) {
        close();
        return false;
    }
    initialized_ = true;
    return true;
}

bool GpuWarpRoi::init_opencl()
{
    cl_platform_id platform = nullptr;
    cl_uint platform_count = 0;
    cl_uint device_count = 0;
    cl_int error = clGetPlatformIDs(1, &platform, &platform_count);
    if (error != CL_SUCCESS || platform_count == 0 || !platform) {
        std::fprintf(stderr, "GpuWarpRoi: clGetPlatformIDs failed (%d)\n", error);
        return false;
    }
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, &device_count);
    if (error != CL_SUCCESS || device_count == 0 || !device_) {
        std::fprintf(stderr, "GpuWarpRoi: clGetDeviceIDs failed (%d)\n", error);
        return false;
    }
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    if (!context_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: clCreateContext failed (%d)\n", error);
        return false;
    }
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &error);
    if (!queue_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: clCreateCommandQueue failed (%d)\n", error);
        return false;
    }
    return true;
}

bool GpuWarpRoi::load_maps()
{
    const std::size_t pixels =
        static_cast<std::size_t>(config_.destination_width) *
        config_.destination_height;
    std::vector<float> map_x;
    std::vector<float> map_y;
    std::vector<unsigned char> valid;
    if (!read_exact(config_.map_x_path, pixels, &map_x) ||
        !read_exact(config_.map_y_path, pixels, &map_y) ||
        !read_exact(config_.valid_path, pixels, &valid))
        return false;

    cl_int error = CL_SUCCESS;
    map_x_cl_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               pixels * sizeof(float), map_x.data(), &error);
    if (!map_x_cl_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: create map_x buffer failed (%d)\n", error);
        return false;
    }
    map_y_cl_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               pixels * sizeof(float), map_y.data(), &error);
    if (!map_y_cl_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: create map_y buffer failed (%d)\n", error);
        return false;
    }
    valid_cl_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               pixels * sizeof(unsigned char), valid.data(), &error);
    if (!valid_cl_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: create valid buffer failed (%d)\n", error);
        return false;
    }
    return true;
}

bool GpuWarpRoi::build_kernels()
{
    cl_int error = CL_SUCCESS;
    program_ = clCreateProgramWithSource(context_, 1, &kKernelSource, nullptr, &error);
    if (!program_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: clCreateProgramWithSource failed (%d)\n", error);
        return false;
    }
    error = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::size_t log_size = 0;
        (void)clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                                    0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        if (log_size)
            (void)clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                                        log_size, log.data(), nullptr);
        std::fprintf(stderr, "GpuWarpRoi: clBuildProgram failed (%d):\n%s\n",
                     error, log.data());
        return false;
    }
    y_kernel_ = clCreateKernel(program_, "warp_y", &error);
    if (!y_kernel_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: create warp_y kernel failed (%d)\n", error);
        return false;
    }
    uv_kernel_ = clCreateKernel(program_, "warp_uv", &error);
    if (!uv_kernel_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: create warp_uv kernel failed (%d)\n", error);
        return false;
    }
    return true;
}

bool GpuWarpRoi::import_buffers(int source_dma_fd)
{
    cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0
    };
    cl_int error = CL_SUCCESS;
    source_cl_ = clImportMemoryARM(context_, CL_MEM_READ_ONLY, properties,
                                   &source_dma_fd, source_size_, &error);
    if (!source_cl_ || error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: import source DMA-BUF failed (%d)\n", error);
        return false;
    }

    destination_fds_.assign(config_.output_buffer_count, -1);
    destination_vas_.assign(config_.output_buffer_count, nullptr);
    destination_cl_.assign(config_.output_buffer_count, nullptr);
    for (int i = 0; i < config_.output_buffer_count; ++i) {
        if (dma_buf_alloc(CMA_HEAP_UNCACHE_PATH, destination_size_,
                          &destination_fds_[i], &destination_vas_[i]) < 0) {
            std::fprintf(stderr,
                         "GpuWarpRoi: allocate destination DMA-BUF %d failed\n", i);
            return false;
        }
        destination_cl_[i] =
            clImportMemoryARM(context_, CL_MEM_READ_WRITE, properties,
                              &destination_fds_[i], destination_size_, &error);
        if (!destination_cl_[i] || error != CL_SUCCESS) {
            std::fprintf(stderr,
                         "GpuWarpRoi: import destination DMA-BUF %d failed (%d)\n",
                         i, error);
            return false;
        }
    }
    return true;
}

bool GpuWarpRoi::set_kernel_arguments()
{
    const int sw = config_.source_width;
    const int sh = config_.source_height;
    const int sys = config_.source_y_stride;
    const int suvs = config_.source_uv_stride;
    const int dw = config_.destination_width;
    const int dh = config_.destination_height;
    const int dys = config_.destination_y_stride;
    const int duvs = config_.destination_uv_stride;

    const cl_int y_errors[] = {
        clSetKernelArg(y_kernel_, 0, sizeof(source_cl_), &source_cl_),
        clSetKernelArg(y_kernel_, 2, sizeof(map_x_cl_), &map_x_cl_),
        clSetKernelArg(y_kernel_, 3, sizeof(map_y_cl_), &map_y_cl_),
        clSetKernelArg(y_kernel_, 4, sizeof(valid_cl_), &valid_cl_),
        clSetKernelArg(y_kernel_, 5, sizeof(sw), &sw),
        clSetKernelArg(y_kernel_, 6, sizeof(sh), &sh),
        clSetKernelArg(y_kernel_, 7, sizeof(sys), &sys),
        clSetKernelArg(y_kernel_, 8, sizeof(dw), &dw),
        clSetKernelArg(y_kernel_, 9, sizeof(dh), &dh),
        clSetKernelArg(y_kernel_, 10, sizeof(dys), &dys),
    };
    for (const cl_int error : y_errors) {
        if (error != CL_SUCCESS) {
            std::fprintf(stderr, "GpuWarpRoi: set warp_y argument failed (%d)\n", error);
            return false;
        }
    }

    const cl_int uv_errors[] = {
        clSetKernelArg(uv_kernel_, 0, sizeof(source_cl_), &source_cl_),
        clSetKernelArg(uv_kernel_, 2, sizeof(map_x_cl_), &map_x_cl_),
        clSetKernelArg(uv_kernel_, 3, sizeof(map_y_cl_), &map_y_cl_),
        clSetKernelArg(uv_kernel_, 4, sizeof(valid_cl_), &valid_cl_),
        clSetKernelArg(uv_kernel_, 5, sizeof(sw), &sw),
        clSetKernelArg(uv_kernel_, 6, sizeof(sh), &sh),
        clSetKernelArg(uv_kernel_, 7, sizeof(sys), &sys),
        clSetKernelArg(uv_kernel_, 8, sizeof(suvs), &suvs),
        clSetKernelArg(uv_kernel_, 9, sizeof(dw), &dw),
        clSetKernelArg(uv_kernel_, 10, sizeof(dh), &dh),
        clSetKernelArg(uv_kernel_, 11, sizeof(dys), &dys),
        clSetKernelArg(uv_kernel_, 12, sizeof(duvs), &duvs),
    };
    for (const cl_int error : uv_errors) {
        if (error != CL_SUCCESS) {
            std::fprintf(stderr, "GpuWarpRoi: set warp_uv argument failed (%d)\n", error);
            return false;
        }
    }
    return true;
}

bool GpuWarpRoi::execute(std::size_t output_index)
{
    if (!initialized_ || !queue_ || !y_kernel_ || !uv_kernel_ ||
        output_index >= destination_fds_.size() ||
        !destination_vas_[output_index] || !destination_cl_[output_index])
        return false;

    std::memset(destination_vas_[output_index], 0,
                static_cast<std::size_t>(config_.destination_y_stride) *
                    config_.destination_height);
    std::memset(
        static_cast<unsigned char *>(destination_vas_[output_index]) +
            static_cast<std::size_t>(config_.destination_y_stride) *
                config_.destination_height,
        128,
        static_cast<std::size_t>(config_.destination_uv_stride) *
            (config_.destination_height / 2));
    if (dma_sync_cpu_to_device(destination_fds_[output_index]) < 0) {
        std::fprintf(stderr, "GpuWarpRoi: destination CPU-to-device sync failed\n");
        return false;
    }
    cl_mem output_cl = destination_cl_[output_index];
    cl_int error = clSetKernelArg(y_kernel_, 1, sizeof(output_cl), &output_cl);
    if (error == CL_SUCCESS)
        error = clSetKernelArg(uv_kernel_, 1, sizeof(output_cl), &output_cl);
    if (error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: set output argument failed (%d)\n", error);
        return false;
    }

    const std::size_t y_global[2] = {
        static_cast<std::size_t>(config_.destination_width),
        static_cast<std::size_t>(config_.destination_height)
    };
    error = clEnqueueNDRangeKernel(queue_, y_kernel_, 2, nullptr, y_global,
                                   nullptr, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: enqueue warp_y failed (%d)\n", error);
        return false;
    }

    const std::size_t uv_global[2] = {
        static_cast<std::size_t>(config_.destination_width / 2),
        static_cast<std::size_t>(config_.destination_height / 2)
    };
    error = clEnqueueNDRangeKernel(queue_, uv_kernel_, 2, nullptr, uv_global,
                                   nullptr, 0, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: enqueue warp_uv failed (%d)\n", error);
        return false;
    }
    error = clFinish(queue_);
    if (error != CL_SUCCESS) {
        std::fprintf(stderr, "GpuWarpRoi: clFinish failed (%d)\n", error);
        return false;
    }
    return true;
}

void GpuWarpRoi::close()
{
    if (queue_)
        (void)clFinish(queue_);
    if (source_cl_) {
        clReleaseMemObject(source_cl_);
        source_cl_ = nullptr;
    }
    for (cl_mem &memory : destination_cl_) {
        if (memory)
            clReleaseMemObject(memory);
        memory = nullptr;
    }
    destination_cl_.clear();
    if (map_x_cl_) {
        clReleaseMemObject(map_x_cl_);
        map_x_cl_ = nullptr;
    }
    if (map_y_cl_) {
        clReleaseMemObject(map_y_cl_);
        map_y_cl_ = nullptr;
    }
    if (valid_cl_) {
        clReleaseMemObject(valid_cl_);
        valid_cl_ = nullptr;
    }
    if (y_kernel_) {
        clReleaseKernel(y_kernel_);
        y_kernel_ = nullptr;
    }
    if (uv_kernel_) {
        clReleaseKernel(uv_kernel_);
        uv_kernel_ = nullptr;
    }
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    device_ = nullptr;
    for (std::size_t i = 0; i < destination_fds_.size(); ++i) {
        if (destination_fds_[i] >= 0)
            dma_buf_free(destination_size_, &destination_fds_[i],
                         destination_vas_[i]);
    }
    destination_fds_.clear();
    destination_vas_.clear();
    source_size_ = 0;
    destination_size_ = 0;
    initialized_ = false;
}

int GpuWarpRoi::output_fd(std::size_t output_index) const
{
    return output_index < destination_fds_.size() ? destination_fds_[output_index] : -1;
}

void *GpuWarpRoi::output_va(std::size_t output_index) const
{
    return output_index < destination_vas_.size() ? destination_vas_[output_index] : nullptr;
}
