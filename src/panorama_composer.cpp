#include "panorama_composer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <rga/im2d.h>

#include "dma_alloc.h"

namespace {

struct InputSpec {
    int width;
    int stride;
    int global_x;
    int body_source_x;
    int body_destination_x;
    int body_width;
};
struct SeamSpec {
    int left;
    int right;
    int x;
    int width;
};
constexpr InputSpec kInputs[4] = {
    {688, 704, 84, 0, 84, 652},
    {436, 448, 738, 34, 772, 366},
    {588, 640, 1140, 34, 1174, 520},
    {640, 640, 1694, 34, 1728, 606},
};
constexpr SeamSpec kSeams[3] = {
    {0, 1, 736, 36}, {1, 2, 1138, 36}, {2, 3, 1694, 34},
};

const char *kKernelSource = R"CLC(
__kernel void blend_y(__global const uchar *l,__global const uchar *r,
 __global uchar *d,__global const float *lw,__global const float *rw,
 __global const uchar *c,int ls,int rs,int ds,int lwidth,int rwidth,
 int lgx,int rgx,int sx,int sw,int h) {
 int x=get_global_id(0),y=get_global_id(1); if(x>=sw||y>=h)return;
 int i=y*sw+x; if(!c[i])return; int gx=sx+x,lx=gx-lgx,rx=gx-rgx;
 float lv=(lx>=0&&lx<lwidth)?(float)l[y*ls+lx]:0.0f;
 float rv=(rx>=0&&rx<rwidth)?(float)r[y*rs+rx]:0.0f;
 d[y*ds+gx]=convert_uchar_sat_rte(lw[i]*lv+rw[i]*rv);
}
__kernel void blend_uv(__global const uchar *l,__global const uchar *r,
 __global uchar *d,__global const float *lw,__global const float *rw,
 __global const uchar *c,int ls,int rs,int ds,int lwidth,int rwidth,
 int lgx,int rgx,int sx,int swuv,int h) {
 int x=get_global_id(0),y=get_global_id(1); if(x>=swuv||y>=h/2)return;
 int i=y*swuv+x; if(!c[i])return; int gx=sx+x*2,lx=gx-lgx,rx=gx-rgx;
 int lb=ls*h,rb=rs*h,db=ds*h; float lu=128,lv=128,ru=128,rv=128;
 if(lx>=0&&lx+1<lwidth){int p=lb+y*ls+lx;lu=l[p];lv=l[p+1];}
 if(rx>=0&&rx+1<rwidth){int p=rb+y*rs+rx;ru=r[p];rv=r[p+1];}
 int o=db+y*ds+gx; d[o]=convert_uchar_sat_rte(lw[i]*lu+rw[i]*ru);
 d[o+1]=convert_uchar_sat_rte(lw[i]*lv+rw[i]*rv);
}
)CLC";

std::vector<unsigned char> read_exact(const std::filesystem::path &path,
                                      std::size_t bytes)
{
    std::vector<unsigned char> data(bytes);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(data.data()),
                static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(stream.gcount()) != bytes ||
        stream.peek() != std::ifstream::traits_type::eof())
        data.clear();
    return data;
}

cl_mem load_buffer(cl_context context, const std::filesystem::path &path,
                   std::size_t bytes)
{
    auto data = read_exact(path, bytes);
    if (data.empty())
        return nullptr;
    cl_int error = CL_SUCCESS;
    return clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytes, data.data(), &error);
}

bool arg(cl_kernel kernel, cl_uint index, std::size_t size, const void *value)
{
    return clSetKernelArg(kernel, index, size, value) == CL_SUCCESS;
}

}  // namespace

PanoramaComposer::PanoramaComposer()
    : context_(nullptr), queue_(nullptr), device_(nullptr), program_(nullptr),
      y_kernel_(nullptr), uv_kernel_(nullptr), output_size_(0), bgr_size_(0),
      initialized_(false)
{
}

PanoramaComposer::~PanoramaComposer() { close(); }

bool PanoramaComposer::init(const std::string &assets_directory)
{
    if (initialized_)
        return false;
    output_size_ = static_cast<std::size_t>(kStride) * kHeight * 3 / 2;
    bgr_size_ =
        static_cast<std::size_t>(kBgrStride) * kBgrHeight * 3;
    if (!init_opencl())
        return false;
    cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0
    };
    for (auto &output : outputs_) {
        if (dma_buf_alloc(CMA_HEAP_UNCACHE_PATH, output_size_,
                          &output.fd, &output.va) < 0)
            return false;
        cl_int error = CL_SUCCESS;
        output.memory = clImportMemoryARM(context_, CL_MEM_READ_WRITE, properties,
                                          &output.fd, output_size_, &error);
        if (!output.memory || error != CL_SUCCESS)
            return false;
    }
    for (auto &output : bgr_outputs_) {
        if (dma_buf_alloc(CMA_HEAP_UNCACHE_PATH, bgr_size_,
                          &output.fd, &output.va) < 0)
            return false;
    }
    if (!load_weights(assets_directory))
        return false;
    initialized_ = true;
    return true;
}

bool PanoramaComposer::init_opencl()
{
    cl_platform_id platform = nullptr;
    cl_uint count = 0;
    cl_int error = clGetPlatformIDs(1, &platform, &count);
    if (error != CL_SUCCESS ||
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, &count) !=
            CL_SUCCESS)
        return false;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &error);
    program_ = clCreateProgramWithSource(context_, 1, &kKernelSource, nullptr, &error);
    if (!context_ || !queue_ || !program_ ||
        clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr) != CL_SUCCESS)
        return false;
    y_kernel_ = clCreateKernel(program_, "blend_y", &error);
    uv_kernel_ = clCreateKernel(program_, "blend_uv", &error);
    return y_kernel_ && uv_kernel_;
}

bool PanoramaComposer::load_weights(const std::string &directory)
{
    for (int i = 0; i < 3; ++i) {
        const auto root = std::filesystem::path(directory);
        const std::string prefix = "seam" + std::to_string(i);
        const std::size_t yc = static_cast<std::size_t>(kSeams[i].width) * kHeight;
        const std::size_t uvc = static_cast<std::size_t>(kSeams[i].width / 2) * (kHeight / 2);
        auto &s = seams_[i];
        s.left_y = load_buffer(context_, root / (prefix + "_left_y.f32"), yc * 4);
        s.right_y = load_buffer(context_, root / (prefix + "_right_y.f32"), yc * 4);
        s.coverage_y = load_buffer(context_, root / (prefix + "_coverage_y.u8"), yc);
        s.left_uv = load_buffer(context_, root / (prefix + "_left_uv.f32"), uvc * 4);
        s.right_uv = load_buffer(context_, root / (prefix + "_right_uv.f32"), uvc * 4);
        s.coverage_uv = load_buffer(context_, root / (prefix + "_coverage_uv.u8"), uvc);
        if (!s.left_y || !s.right_y || !s.coverage_y ||
            !s.left_uv || !s.right_uv || !s.coverage_uv)
            return false;
    }
    return true;
}

cl_mem PanoramaComposer::import_input(int fd, std::size_t size)
{
    const auto found = input_cache_.find(fd);
    if (found != input_cache_.end())
        return found->second;
    cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0
    };
    cl_int error = CL_SUCCESS;
    cl_mem memory = clImportMemoryARM(context_, CL_MEM_READ_ONLY, properties,
                                      &fd, size, &error);
    if (!memory || error != CL_SUCCESS)
        return nullptr;
    input_cache_[fd] = memory;
    return memory;
}

bool PanoramaComposer::run_rga_body(
    const std::array<WarpedFrameRef, 4> &frames, std::size_t slot)
{
    auto &output = outputs_[slot];
    std::memset(output.va, 0, static_cast<std::size_t>(kStride) * kHeight);
    std::memset(static_cast<unsigned char *>(output.va) +
                    static_cast<std::size_t>(kStride) * kHeight,
                128, static_cast<std::size_t>(kStride) * kHeight / 2);
    if (dma_sync_cpu_to_device(output.fd) < 0)
        return false;
    rga_buffer_t destination = wrapbuffer_fd_t(
        output.fd, kWidth, kHeight, kStride, kHeight, RK_FORMAT_YCbCr_420_SP);
    for (int i = 0; i < 4; ++i) {
        if (!frames[i].valid() || frames[i].width() != kInputs[i].width ||
            frames[i].stride() != kInputs[i].stride)
            return false;
        rga_buffer_t source = wrapbuffer_fd_t(
            frames[i].fd(), frames[i].width(), kHeight, frames[i].stride(),
            kHeight, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t pattern{};
        im_rect sr{kInputs[i].body_source_x, 0, kInputs[i].body_width, kHeight};
        im_rect dr{kInputs[i].body_destination_x, 0, kInputs[i].body_width, kHeight};
        im_rect pr{};
        if (improcess(source, destination, pattern, sr, dr, pr, IM_SYNC) !=
            IM_STATUS_SUCCESS)
            return false;
    }
    return true;
}

bool PanoramaComposer::run_gpu_seams(
    const std::array<WarpedFrameRef, 4> &frames, std::size_t slot)
{
    cl_mem sources[4] = {};
    for (int i = 0; i < 4; ++i) {
        const std::size_t size =
            static_cast<std::size_t>(frames[i].stride()) * kHeight * 3 / 2;
        sources[i] = import_input(frames[i].fd(), size);
        if (!sources[i])
            return false;
    }
    for (int i = 0; i < 3; ++i) {
        const auto &spec = kSeams[i];
        const auto &left = kInputs[spec.left];
        const auto &right = kInputs[spec.right];
        auto &s = seams_[i];
        cl_uint n = 0;
        bool ok =
            arg(y_kernel_, n++, sizeof(sources[spec.left]), &sources[spec.left]) &&
            arg(y_kernel_, n++, sizeof(sources[spec.right]), &sources[spec.right]) &&
            arg(y_kernel_, n++, sizeof(outputs_[slot].memory), &outputs_[slot].memory) &&
            arg(y_kernel_, n++, sizeof(s.left_y), &s.left_y) &&
            arg(y_kernel_, n++, sizeof(s.right_y), &s.right_y) &&
            arg(y_kernel_, n++, sizeof(s.coverage_y), &s.coverage_y) &&
            arg(y_kernel_, n++, sizeof(left.stride), &left.stride) &&
            arg(y_kernel_, n++, sizeof(right.stride), &right.stride) &&
            arg(y_kernel_, n++, sizeof(kStride), &kStride) &&
            arg(y_kernel_, n++, sizeof(left.width), &left.width) &&
            arg(y_kernel_, n++, sizeof(right.width), &right.width) &&
            arg(y_kernel_, n++, sizeof(left.global_x), &left.global_x) &&
            arg(y_kernel_, n++, sizeof(right.global_x), &right.global_x) &&
            arg(y_kernel_, n++, sizeof(spec.x), &spec.x) &&
            arg(y_kernel_, n++, sizeof(spec.width), &spec.width) &&
            arg(y_kernel_, n++, sizeof(kHeight), &kHeight);
        const std::size_t gy[2] = {
            static_cast<std::size_t>(spec.width), static_cast<std::size_t>(kHeight)};
        if (!ok || clEnqueueNDRangeKernel(queue_, y_kernel_, 2, nullptr, gy,
                                           nullptr, 0, nullptr, nullptr) != CL_SUCCESS)
            return false;
        const int width_uv = spec.width / 2;
        n = 0;
        ok =
            arg(uv_kernel_, n++, sizeof(sources[spec.left]), &sources[spec.left]) &&
            arg(uv_kernel_, n++, sizeof(sources[spec.right]), &sources[spec.right]) &&
            arg(uv_kernel_, n++, sizeof(outputs_[slot].memory), &outputs_[slot].memory) &&
            arg(uv_kernel_, n++, sizeof(s.left_uv), &s.left_uv) &&
            arg(uv_kernel_, n++, sizeof(s.right_uv), &s.right_uv) &&
            arg(uv_kernel_, n++, sizeof(s.coverage_uv), &s.coverage_uv) &&
            arg(uv_kernel_, n++, sizeof(left.stride), &left.stride) &&
            arg(uv_kernel_, n++, sizeof(right.stride), &right.stride) &&
            arg(uv_kernel_, n++, sizeof(kStride), &kStride) &&
            arg(uv_kernel_, n++, sizeof(left.width), &left.width) &&
            arg(uv_kernel_, n++, sizeof(right.width), &right.width) &&
            arg(uv_kernel_, n++, sizeof(left.global_x), &left.global_x) &&
            arg(uv_kernel_, n++, sizeof(right.global_x), &right.global_x) &&
            arg(uv_kernel_, n++, sizeof(spec.x), &spec.x) &&
            arg(uv_kernel_, n++, sizeof(width_uv), &width_uv) &&
            arg(uv_kernel_, n++, sizeof(kHeight), &kHeight);
        const std::size_t guv[2] = {
            static_cast<std::size_t>(width_uv), static_cast<std::size_t>(kHeight / 2)};
        if (!ok || clEnqueueNDRangeKernel(queue_, uv_kernel_, 2, nullptr, guv,
                                           nullptr, 0, nullptr, nullptr) != CL_SUCCESS)
            return false;
    }
    return clFinish(queue_) == CL_SUCCESS;
}

bool PanoramaComposer::compose(
    const std::array<WarpedFrameRef, 4> &frames, std::size_t slot)
{
    if (!initialized_ || slot >= kOutputSlots)
        return false;
    const auto begin = std::chrono::steady_clock::now();
    if (!run_rga_body(frames, slot))
        return false;
    const auto after_rga = std::chrono::steady_clock::now();
    if (!run_gpu_seams(frames, slot))
        return false;
    const auto after_gpu = std::chrono::steady_clock::now();
    stats_.rga_body_total_ms +=
        std::chrono::duration<double, std::milli>(after_rga - begin).count();
    stats_.gpu_seam_total_ms +=
        std::chrono::duration<double, std::milli>(after_gpu - after_rga).count();
    ++stats_.frames;
    return true;
}

bool PanoramaComposer::convert_bgr(std::size_t slot, std::size_t bgr_slot)
{
    if (!initialized_ || slot >= kOutputSlots || bgr_slot >= kBgrSlots)
        return false;
    const auto begin = std::chrono::steady_clock::now();
    rga_buffer_t source = wrapbuffer_fd_t(
        outputs_[slot].fd, kWidth, kHeight, kStride, kHeight,
        RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t destination = wrapbuffer_fd_t(
        bgr_outputs_[bgr_slot].fd, kBgrWidth, kBgrHeight,
        kBgrStride, kBgrHeight,
        RK_FORMAT_BGR_888);
    rga_buffer_t pattern{};
    im_rect source_rect{
        kBgrCropX, kBgrCropY, kBgrWidth, kBgrHeight,
    };
    im_rect destination_rect{
        0, 0, kBgrWidth, kBgrHeight,
    };
    im_rect pattern_rect{};
    if (improcess(source, destination, pattern, source_rect, destination_rect,
                  pattern_rect, IM_SYNC) != IM_STATUS_SUCCESS ||
        dma_sync_device_to_cpu(bgr_outputs_[bgr_slot].fd) < 0)
        return false;
    stats_.bgr_convert_total_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    return true;
}

int PanoramaComposer::bgr_fd(std::size_t slot) const
{
    return slot < bgr_outputs_.size() ? bgr_outputs_[slot].fd : -1;
}

void *PanoramaComposer::bgr_va(std::size_t slot) const
{
    return slot < bgr_outputs_.size() ? bgr_outputs_[slot].va : nullptr;
}

int PanoramaComposer::output_fd(std::size_t slot) const
{
    return slot < outputs_.size() ? outputs_[slot].fd : -1;
}
void *PanoramaComposer::output_va(std::size_t slot) const
{
    return slot < outputs_.size() ? outputs_[slot].va : nullptr;
}

void PanoramaComposer::close()
{
    if (queue_)
        clFinish(queue_);
    for (auto &entry : input_cache_)
        clReleaseMemObject(entry.second);
    input_cache_.clear();
    for (auto &s : seams_) {
        cl_mem *items[] = {&s.left_y, &s.right_y, &s.coverage_y,
                           &s.left_uv, &s.right_uv, &s.coverage_uv};
        for (cl_mem *item : items) {
            if (*item) clReleaseMemObject(*item);
            *item = nullptr;
        }
    }
    for (auto &output : outputs_) {
        if (output.memory) clReleaseMemObject(output.memory);
        output.memory = nullptr;
    }
    if (y_kernel_) clReleaseKernel(y_kernel_);
    if (uv_kernel_) clReleaseKernel(uv_kernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    y_kernel_ = uv_kernel_ = nullptr; program_ = nullptr; queue_ = nullptr;
    context_ = nullptr; device_ = nullptr;
    for (auto &output : bgr_outputs_) {
        if (output.fd >= 0)
            dma_buf_free(bgr_size_, &output.fd, output.va);
        output.fd = -1;
        output.va = nullptr;
    }
    for (auto &output : outputs_) {
        if (output.fd >= 0)
            dma_buf_free(output_size_, &output.fd, output.va);
        output.fd = -1; output.va = nullptr;
    }
    initialized_ = false;
}
