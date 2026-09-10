#include "trigger_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace
{
constexpr int camera_count = 3;

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__constant__ float gaussian_weights[64];

__device__ float read_float(const float* data, std::size_t step, int width, int height,
                            int camera, int y, int x)
{
    if (camera < 0 || camera >= camera_count || y < 0 || y >= height || x < 0 || x >= width)
        return 0.0f;
    const auto* row = reinterpret_cast<const float*>(
        reinterpret_cast<const unsigned char*>(data) + (camera * height + y) * step);
    return row[x];
}

__device__ uint16_t read_uint16(const uint16_t* data, std::size_t step, int width,
                                int height, int camera, int y, int x)
{
    if (camera < 0 || camera >= camera_count || y < 0 || y >= height || x < 0 || x >= width)
        return 0;
    const auto* row = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const unsigned char*>(data) + (camera * height + y) * step);
    return row[x];
}

__device__ unsigned char read_mask(const unsigned char* data, std::size_t step,
                                   int width, int height, int camera, int y, int x)
{
    if (camera < 0 || camera >= camera_count || y < 0 || y >= height || x < 0 || x >= width)
        return 0;
    const auto* row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(data) + (camera * height + y) * step);
    return row[x];
}

__global__ void subtract_kernel(const uint16_t* image, std::size_t image_step,
                                const float* pedestal, std::size_t pedestal_step,
                                float* output, std::size_t output_step,
                                int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = static_cast<float>(read_uint16(image, image_step, width, height, camera, y, x))
        - read_float(pedestal, pedestal_step, width, height, camera, y, x);
}

__global__ void laplacian_kernel(const float* source, std::size_t source_step,
                                 float* output, std::size_t output_step,
                                 int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    float value = 8.0f * read_float(source, source_step, width, height, camera, y, x);
    for (int offset_y = -1; offset_y <= 1; ++offset_y)
        for (int offset_x = -1; offset_x <= 1; ++offset_x)
            if (offset_x != 0 || offset_y != 0)
                value -= read_float(source, source_step, width, height, camera,
                                    y + offset_y, x + offset_x);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = value;
}

__global__ void threshold_kernel(const float* source, std::size_t source_step,
                                 unsigned char* output, std::size_t output_step,
                                 float cut, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    auto* row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = read_float(source, source_step, width, height, camera, y, x) >= cut ? 255 : 0;
}

__global__ void zero_mask_kernel(float* image, std::size_t image_step,
                                 const unsigned char* mask, std::size_t mask_step,
                                 int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    if (read_mask(mask, mask_step, width, height, camera, y, x) != 0)
    {
        auto* row = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(image) + (camera * height + y) * image_step);
        row[x] = 0.0f;
    }
}

__global__ void dilate_horizontal_kernel(const unsigned char* source, std::size_t source_step,
                                         unsigned char* output, std::size_t output_step,
                                         int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    unsigned char value = 0;
    for (int offset = -radius; offset <= radius; ++offset)
        value = max(value, read_mask(source, source_step, width, height, camera, y, x + offset));
    auto* row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = value;
}

__global__ void dilate_vertical_kernel(const unsigned char* source, std::size_t source_step,
                                       unsigned char* output, std::size_t output_step,
                                       int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    unsigned char value = 0;
    for (int offset = -radius; offset <= radius; ++offset)
        value = max(value, read_mask(source, source_step, width, height, camera, y + offset, x));
    auto* row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = value;
}

__global__ void gaussian_horizontal_kernel(const float* source, std::size_t source_step,
                                           float* output, std::size_t output_step,
                                           int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
        value += gaussian_weights[offset + radius]
            * read_float(source, source_step, width, height, camera, y, x + offset);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = value;
}

__global__ void gaussian_vertical_kernel(const float* source, std::size_t source_step,
                                         float* output, std::size_t output_step,
                                         int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
        value += gaussian_weights[offset + radius]
            * read_float(source, source_step, width, height, camera, y + offset, x);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    row[x] = value;
}

__global__ void apply_mask_kernel(const uint16_t* image, std::size_t image_step,
                                  const unsigned char* mask, std::size_t mask_step,
                                  uint16_t* output, std::size_t output_step,
                                  int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    const auto* image_row = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const unsigned char*>(image) + (camera * height + y) * image_step);
    const auto* mask_row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(mask) + (camera * height + y) * mask_step);
    auto* output_row = reinterpret_cast<uint16_t*>(
        reinterpret_cast<unsigned char*>(output) + (camera * height + y) * output_step);
    output_row[x] = mask_row[x] != 0 ? image_row[x] : 0;
}

__global__ void count_mask_kernel(const unsigned char* mask, std::size_t mask_step,
                                  unsigned int* counts, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(camera_count) * width * height;
    if (index >= count)
        return;
    const int camera = static_cast<int>(index / (width * height));
    const std::size_t local = index % (width * height);
    const int y = static_cast<int>(local / width);
    const int x = static_cast<int>(local % width);
    const auto* row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(mask) + (camera * height + y) * mask_step);
    if (row[x] != 0)
        atomicAdd(&counts[camera], 1U);
}

int blocks_for(int width, int height)
{
    constexpr int block_size = 256;
    return (camera_count * width * height + block_size - 1) / block_size;
}
}

Image::Image(int image_width, int image_height)
    : width(image_width), height(image_height),
      data(static_cast<std::size_t>(image_width) * image_height)
{
}

FloatImage::FloatImage(int image_width, int image_height)
    : width(image_width), height(image_height),
      data(static_cast<std::size_t>(image_width) * image_height)
{
}

struct BatchTrigger::Impl
{
    int width;
    int height;
    int gaussian_radius;
    float spark_cut;
    float threshold_cut;
    int dilation_radius;
    cv::cuda::Stream stream;
    cv::cuda::GpuMat input_gpu;
    cv::cuda::GpuMat result_gpu;
        cv::cuda::GpuMat counts_gpu;
    cv::cuda::GpuMat pedestal_gpu;
    cv::cuda::GpuMat pedestal_subtracted;
    cv::cuda::GpuMat laplacian;
    cv::cuda::GpuMat spark_mask;
    cv::cuda::GpuMat spark_dilated_horizontal;
    cv::cuda::GpuMat spark_dilated;
    cv::cuda::GpuMat sparkless;
    cv::cuda::GpuMat gaussian_horizontal;
    cv::cuda::GpuMat filtered;
    cv::cuda::GpuMat centroid_mask;
    cv::cuda::GpuMat centroid_dilated_horizontal;
    cv::cuda::GpuMat centroid_dilated;
    uint16_t* input_host = nullptr;
    uint16_t* output_host = nullptr;
        unsigned int counts_host[camera_count]{};
    std::vector<float> pedestal_host;
    static constexpr int stage_count = 11;
    cudaEvent_t stage_start[stage_count]{};
    cudaEvent_t stage_end[stage_count]{};
    bool pedestals_set = false;
    bool pending = false;

    Impl(int image_width, int image_height, int kernel_size, float sigma,
         float spark, float threshold, int radius)
        : width(image_width), height(image_height), gaussian_radius(kernel_size / 2),
          spark_cut(spark), threshold_cut(threshold), dilation_radius(radius),
          input_gpu(camera_count * image_height, image_width, CV_16U),
          result_gpu(camera_count * image_height, image_width, CV_16U),
                  counts_gpu(1, camera_count, CV_32S),
          pedestal_gpu(camera_count * image_height, image_width, CV_32F),
          pedestal_subtracted(camera_count * image_height, image_width, CV_32F),
          laplacian(camera_count * image_height, image_width, CV_32F),
          spark_mask(camera_count * image_height, image_width, CV_8U),
          spark_dilated_horizontal(camera_count * image_height, image_width, CV_8U),
          spark_dilated(camera_count * image_height, image_width, CV_8U),
          sparkless(camera_count * image_height, image_width, CV_32F),
          gaussian_horizontal(camera_count * image_height, image_width, CV_32F),
          filtered(camera_count * image_height, image_width, CV_32F),
          centroid_mask(camera_count * image_height, image_width, CV_8U),
          centroid_dilated_horizontal(camera_count * image_height, image_width, CV_8U),
          centroid_dilated(camera_count * image_height, image_width, CV_8U),
          pedestal_host(static_cast<std::size_t>(camera_count) * image_width * image_height)
    {
        if (kernel_size <= 0 || kernel_size % 2 == 0 || sigma <= 0.0f || radius < 0)
            throw std::runtime_error("Invalid batch trigger parameters");
        check_cuda(cudaMallocHost(
            reinterpret_cast<void**>(&input_host),
            static_cast<std::size_t>(camera_count) * image_width * image_height
                * sizeof(uint16_t)), "allocate batch pinned input");
        check_cuda(cudaMallocHost(
            reinterpret_cast<void**>(&output_host),
            static_cast<std::size_t>(camera_count) * image_width * image_height
                * sizeof(uint16_t)), "allocate batch pinned output");

        std::vector<float> weights(kernel_size);
        const float denominator = 2.0f * sigma * sigma;
        float sum = 0.0f;
        for (int index = 0; index < kernel_size; ++index)
        {
            const int offset = index - gaussian_radius;
            weights[index] = std::exp(-(offset * offset) / denominator);
            sum += weights[index];
        }
        for (float& weight : weights)
            weight /= sum;
        check_cuda(cudaMemcpyToSymbol(gaussian_weights, weights.data(),
                                      weights.size() * sizeof(float)),
                   "copy Gaussian weights");
        for (int stage = 0; stage < stage_count; ++stage)
        {
            check_cuda(cudaEventCreate(&stage_start[stage]), "create batch stage start");
            check_cuda(cudaEventCreate(&stage_end[stage]), "create batch stage end");
        }
    }

    ~Impl()
    {
        if (input_host)
            cudaFreeHost(input_host);
        if (output_host)
            cudaFreeHost(output_host);
        for (int stage = 0; stage < stage_count; ++stage)
        {
            if (stage_start[stage])
                cudaEventDestroy(stage_start[stage]);
            if (stage_end[stage])
                cudaEventDestroy(stage_end[stage]);
        }
    }
};

BatchTrigger::BatchTrigger(int width, int height, int gaussian_kernel_size,
                           float gaussian_sigma, float spark_cut,
                           float threshold_cut, int dilation_radius)
    : impl_(new Impl(width, height, gaussian_kernel_size, gaussian_sigma,
                     spark_cut, threshold_cut, dilation_radius))
{
}

BatchTrigger::~BatchTrigger() = default;

void BatchTrigger::set_pedestals(const std::array<FloatImage, 3>& pedestals)
{
    for (int camera = 0; camera < camera_count; ++camera)
    {
        if (pedestals[camera].width != impl_->width || pedestals[camera].height != impl_->height)
            throw std::runtime_error("Batch pedestal dimensions do not match");
        std::copy(pedestals[camera].data.begin(), pedestals[camera].data.end(),
                  impl_->pedestal_host.begin() + camera * impl_->width * impl_->height);
    }
    cv::Mat host(camera_count * impl_->height, impl_->width, CV_32F,
                 impl_->pedestal_host.data());
    impl_->pedestal_gpu.upload(host);
    impl_->pedestals_set = true;
}

void BatchTrigger::enqueue(const std::array<Image, 3>& images)
{
    if (!impl_->pedestals_set)
        throw std::runtime_error("Batch pedestals must be set before enqueue");
    if (impl_->pending)
        throw std::runtime_error("Batch trigger already has a pending image set");

    for (int camera = 0; camera < camera_count; ++camera)
    {
        if (images[camera].width != impl_->width || images[camera].height != impl_->height)
            throw std::runtime_error("Batch image dimensions do not match");
        std::transform(images[camera].data.begin(), images[camera].data.end(),
                   impl_->input_host + camera * impl_->width * impl_->height,
                       [](int pixel) { return static_cast<uint16_t>(pixel); });
    }
    cv::Mat input_host(camera_count * impl_->height, impl_->width, CV_16U,
                   impl_->input_host);
    auto record_start = [&](int stage) {
        check_cuda(cudaEventRecord(impl_->stage_start[stage],
                                   reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
                   "record batch stage start");
    };
    auto record_end = [&](int stage) {
        check_cuda(cudaEventRecord(impl_->stage_end[stage],
                                   reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
                   "record batch stage end");
    };
    record_start(0);
    impl_->input_gpu.upload(input_host, impl_->stream);
    record_end(0);

    const int blocks = blocks_for(impl_->width, impl_->height);
    constexpr int block_size = 256;
    auto launch = [&](const void* kernel, void** arguments, const char* name) {
        check_cuda(cudaLaunchKernel(kernel, dim3(blocks), dim3(block_size), arguments, 0,
                                    reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())), name);
    };

    int width = impl_->width;
    int height = impl_->height;
    int gaussian_radius = impl_->gaussian_radius;
    int spark_radius = 1;
    int dilation_radius = impl_->dilation_radius;
    float spark_cut = impl_->spark_cut;
    float threshold_cut = impl_->threshold_cut;
    uint16_t* input_pointer = impl_->input_gpu.ptr<uint16_t>();
    float* pedestal_pointer = impl_->pedestal_gpu.ptr<float>();
    float* pedestal_subtracted_pointer = impl_->pedestal_subtracted.ptr<float>();
    float* laplacian_pointer = impl_->laplacian.ptr<float>();
    unsigned char* spark_pointer = impl_->spark_mask.ptr<unsigned char>();
    unsigned char* spark_horizontal_pointer = impl_->spark_dilated_horizontal.ptr<unsigned char>();
    unsigned char* spark_dilated_pointer = impl_->spark_dilated.ptr<unsigned char>();
    float* sparkless_pointer = impl_->sparkless.ptr<float>();
    float* gaussian_horizontal_pointer = impl_->gaussian_horizontal.ptr<float>();
    float* filtered_pointer = impl_->filtered.ptr<float>();
    unsigned char* centroid_pointer = impl_->centroid_mask.ptr<unsigned char>();
    unsigned char* centroid_horizontal_pointer = impl_->centroid_dilated_horizontal.ptr<unsigned char>();
    unsigned char* centroid_dilated_pointer = impl_->centroid_dilated.ptr<unsigned char>();
    uint16_t* result_pointer = impl_->result_gpu.ptr<uint16_t>();

    std::size_t input_step = impl_->input_gpu.step;
    std::size_t pedestal_step = impl_->pedestal_gpu.step;
    std::size_t pedestal_subtracted_step = impl_->pedestal_subtracted.step;
    std::size_t laplacian_step = impl_->laplacian.step;
    std::size_t spark_step = impl_->spark_mask.step;
    std::size_t spark_horizontal_step = impl_->spark_dilated_horizontal.step;
    std::size_t spark_dilated_step = impl_->spark_dilated.step;
    std::size_t sparkless_step = impl_->sparkless.step;
    std::size_t gaussian_horizontal_step = impl_->gaussian_horizontal.step;
    std::size_t filtered_step = impl_->filtered.step;
    std::size_t centroid_step = impl_->centroid_mask.step;
    std::size_t centroid_horizontal_step = impl_->centroid_dilated_horizontal.step;
    std::size_t centroid_dilated_step = impl_->centroid_dilated.step;
    std::size_t result_step = impl_->result_gpu.step;

    record_start(1);
    void* subtract_arguments[] = {&input_pointer, &input_step, &pedestal_pointer,
        &pedestal_step, &pedestal_subtracted_pointer, &pedestal_subtracted_step,
        &width, &height};
    launch(reinterpret_cast<const void*>(subtract_kernel), subtract_arguments, "batch subtract");
    record_end(1);
    record_start(2);
    void* laplacian_arguments[] = {&pedestal_subtracted_pointer, &pedestal_subtracted_step,
        &laplacian_pointer, &laplacian_step, &width, &height};
    launch(reinterpret_cast<const void*>(laplacian_kernel), laplacian_arguments, "batch Laplacian");
    record_end(2);
    record_start(3);
    void* threshold_arguments[] = {&laplacian_pointer, &laplacian_step, &spark_pointer,
        &spark_step, &spark_cut, &width, &height};
    launch(reinterpret_cast<const void*>(threshold_kernel), threshold_arguments, "batch spark threshold");
    record_end(3);
    record_start(4);
    void* spark_horizontal_arguments[] = {&spark_pointer, &spark_step,
        &spark_horizontal_pointer, &spark_horizontal_step, &spark_radius, &width, &height};
    launch(reinterpret_cast<const void*>(dilate_horizontal_kernel), spark_horizontal_arguments,
           "batch spark horizontal dilation");
    void* spark_vertical_arguments[] = {&spark_horizontal_pointer, &spark_horizontal_step,
        &spark_dilated_pointer, &spark_dilated_step, &spark_radius, &width, &height};
    launch(reinterpret_cast<const void*>(dilate_vertical_kernel), spark_vertical_arguments,
           "batch spark vertical dilation");
    record_end(4);
    record_start(5);
    impl_->pedestal_subtracted.copyTo(impl_->sparkless, impl_->stream);
    sparkless_pointer = impl_->sparkless.ptr<float>();
    void* zero_arguments[] = {&sparkless_pointer, &sparkless_step,
        &spark_dilated_pointer, &spark_dilated_step, &width, &height};
    launch(reinterpret_cast<const void*>(zero_mask_kernel), zero_arguments, "batch spark mask");
    record_end(5);
    record_start(6);
    void* gaussian_horizontal_arguments[] = {&sparkless_pointer,
        &sparkless_step, &gaussian_horizontal_pointer, &gaussian_horizontal_step,
        &gaussian_radius, &width, &height};
    launch(reinterpret_cast<const void*>(gaussian_horizontal_kernel), gaussian_horizontal_arguments,
           "batch Gaussian horizontal");
    void* gaussian_vertical_arguments[] = {&gaussian_horizontal_pointer,
        &gaussian_horizontal_step, &filtered_pointer, &filtered_step, &gaussian_radius,
        &width, &height};
    launch(reinterpret_cast<const void*>(gaussian_vertical_kernel), gaussian_vertical_arguments,
           "batch Gaussian vertical");
    record_end(6);
    record_start(7);
    void* centroid_arguments[] = {&filtered_pointer, &filtered_step, &centroid_pointer,
        &centroid_step, &threshold_cut, &width, &height};
    launch(reinterpret_cast<const void*>(threshold_kernel), centroid_arguments,
           "batch centroid threshold");
    record_end(7);
    record_start(8);
    void* centroid_horizontal_arguments[] = {&centroid_pointer, &centroid_step,
        &centroid_horizontal_pointer, &centroid_horizontal_step, &dilation_radius,
        &width, &height};
    launch(reinterpret_cast<const void*>(dilate_horizontal_kernel), centroid_horizontal_arguments,
           "batch centroid horizontal dilation");
    void* centroid_vertical_arguments[] = {&centroid_horizontal_pointer,
        &centroid_horizontal_step, &centroid_dilated_pointer, &centroid_dilated_step,
        &dilation_radius, &width, &height};
    launch(reinterpret_cast<const void*>(dilate_vertical_kernel), centroid_vertical_arguments,
           "batch centroid vertical dilation");
    record_end(8);
    record_start(9);
    void* result_arguments[] = {&input_pointer, &input_step,
        &centroid_dilated_pointer, &centroid_dilated_step, &result_pointer,
        &result_step, &width, &height};
    launch(reinterpret_cast<const void*>(apply_mask_kernel), result_arguments,
           "batch final GPU image");
    check_cuda(cudaMemsetAsync(impl_->counts_gpu.data, 0,
                               camera_count * sizeof(unsigned int),
                               reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
               "clear batch mask counts");
    unsigned char* mask_pointer = impl_->centroid_dilated.ptr<unsigned char>();
    unsigned int* counts_pointer = impl_->counts_gpu.ptr<unsigned int>();
    void* count_arguments[] = {&mask_pointer, &centroid_dilated_step,
        &counts_pointer, &width, &height};
    launch(reinterpret_cast<const void*>(count_mask_kernel), count_arguments,
           "batch count final mask");
    record_end(9);
    impl_->pending = true;
}

void BatchTrigger::synchronize(std::array<Image, 3>& results, BatchTiming& timing)
{
    if (!impl_->pending)
        throw std::runtime_error("Batch trigger has no pending image set");
    cv::Mat result_host(impl_->height * camera_count, impl_->width, CV_16U,
                        impl_->output_host);
        cv::Mat counts_host(1, camera_count, CV_32S);
    check_cuda(cudaEventRecord(impl_->stage_start[10],
                               reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
               "record batch download start");
    impl_->result_gpu.download(result_host, impl_->stream);
        impl_->counts_gpu.download(counts_host, impl_->stream);
    check_cuda(cudaEventRecord(impl_->stage_end[10],
                               reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
               "record batch download end");
    impl_->stream.waitForCompletion();
    auto elapsed_stage = [&](int stage) {
        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, impl_->stage_start[stage],
                                        impl_->stage_end[stage]),
                   "measure batch stage");
        return static_cast<double>(milliseconds);
    };
    timing.upload_ms = elapsed_stage(0);
    timing.pedestal_ms = elapsed_stage(1);
    timing.laplacian_ms = elapsed_stage(2);
    timing.spark_threshold_ms = elapsed_stage(3);
    timing.spark_dilation_ms = elapsed_stage(4);
    timing.spark_mask_ms = elapsed_stage(5);
    timing.gaussian_ms = elapsed_stage(6);
    timing.centroid_threshold_ms = elapsed_stage(7);
    timing.centroid_dilation_ms = elapsed_stage(8);
    timing.mask_apply_ms = elapsed_stage(9);
    timing.download_ms = elapsed_stage(10);
    for (int camera = 0; camera < camera_count; ++camera)
    {
        results[camera] = Image(impl_->width, impl_->height);
        const std::size_t image_offset = static_cast<std::size_t>(camera)
            * impl_->width * impl_->height;
        cv::Mat source(impl_->height, impl_->width, CV_16U,
                       impl_->input_host + image_offset);
        cv::Mat result(impl_->height, impl_->width, CV_32S,
                       results[camera].data.data());
        cv::Mat camera_result = result_host.rowRange(
            camera * impl_->height, (camera + 1) * impl_->height);
        camera_result.convertTo(result, CV_32S);
        timing.triggered_pixels[camera] = static_cast<std::size_t>(
            counts_host.at<int>(0, camera));
    }
    float full_elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&full_elapsed_ms, impl_->stage_start[0],
                                    impl_->stage_end[10]), "measure batch full trigger span");
    timing.full_trigger_ms = static_cast<double>(full_elapsed_ms);
    impl_->pending = false;
}

void BatchTrigger::synchronize_output(BatchTiming& timing)
{
    if (!impl_->pending)
        throw std::runtime_error("Batch trigger has no pending image set");
    cv::Mat result_host(impl_->height * camera_count, impl_->width, CV_16U,
                        impl_->output_host);
    cv::Mat counts_host(1, camera_count, CV_32S);
    check_cuda(cudaEventRecord(impl_->stage_start[10],
                               reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
               "record batch download start");
    impl_->result_gpu.download(result_host, impl_->stream);
    impl_->counts_gpu.download(counts_host, impl_->stream);
    check_cuda(cudaEventRecord(impl_->stage_end[10],
                               reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
               "record batch download end");
    impl_->stream.waitForCompletion();
    auto elapsed_stage = [&](int stage) {
        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, impl_->stage_start[stage],
                                        impl_->stage_end[stage]), "measure batch stage");
        return static_cast<double>(milliseconds);
    };
    timing.upload_ms = elapsed_stage(0);
    timing.pedestal_ms = elapsed_stage(1);
    timing.laplacian_ms = elapsed_stage(2);
    timing.spark_threshold_ms = elapsed_stage(3);
    timing.spark_dilation_ms = elapsed_stage(4);
    timing.spark_mask_ms = elapsed_stage(5);
    timing.gaussian_ms = elapsed_stage(6);
    timing.centroid_threshold_ms = elapsed_stage(7);
    timing.centroid_dilation_ms = elapsed_stage(8);
    timing.mask_apply_ms = elapsed_stage(9);
    timing.download_ms = elapsed_stage(10);
    for (int camera = 0; camera < camera_count; ++camera)
    {
        timing.triggered_pixels[camera] = static_cast<std::size_t>(
            counts_host.at<int>(0, camera));
    }
    float full_elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&full_elapsed_ms, impl_->stage_start[0],
                                    impl_->stage_end[10]), "measure batch full trigger span");
    timing.full_trigger_ms = static_cast<double>(full_elapsed_ms);
    impl_->pending = false;
}

void BatchTrigger::write_output_pgm(const std::string& filename, int camera) const
{
    if (camera < 0 || camera >= camera_count)
        throw std::runtime_error("Invalid camera index");
    std::ofstream output(filename, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create PGM: " + filename);
    output << "P5\n" << impl_->width << ' ' << impl_->height << "\n65535\n";
    const auto* pixels = impl_->output_host
        + static_cast<std::size_t>(camera) * impl_->width * impl_->height;
    const std::size_t count = static_cast<std::size_t>(impl_->width) * impl_->height;
    for (std::size_t index = 0; index < count; ++index)
    {
        output.put(static_cast<char>(pixels[index] >> 8));
        output.put(static_cast<char>(pixels[index] & 0xff));
    }
}

void BatchTrigger::copy_centroid_mask(int camera, std::vector<uint8_t>& output)
{
    if (camera < 0 || camera >= camera_count)
        throw std::runtime_error("Invalid camera index");
    cv::Mat mask;
    impl_->centroid_mask.download(mask);
    output.resize(static_cast<std::size_t>(impl_->width) * impl_->height);
    for (int y = 0; y < impl_->height; ++y)
    {
        const auto* row = mask.ptr<unsigned char>(camera * impl_->height + y);
        std::copy(row, row + impl_->width,
                  output.begin() + static_cast<std::size_t>(y) * impl_->width);
    }
}

void BatchTrigger::copy_filtered_image(int camera, std::vector<float>& output)
{
    if (camera < 0 || camera >= camera_count)
        throw std::runtime_error("Invalid camera index");
    cv::Mat filtered;
    impl_->filtered.download(filtered);
    output.resize(static_cast<std::size_t>(impl_->width) * impl_->height);
    for (int y = 0; y < impl_->height; ++y)
    {
        const auto* row = filtered.ptr<float>(camera * impl_->height + y);
        std::copy(row, row + impl_->width,
                  output.begin() + static_cast<std::size_t>(y) * impl_->width);
    }
}

void BatchTrigger::copy_sparkless_image(int camera, std::vector<float>& output)
{
    if (camera < 0 || camera >= camera_count)
        throw std::runtime_error("Invalid camera index");
    cv::Mat sparkless;
    impl_->sparkless.download(sparkless);
    output.resize(static_cast<std::size_t>(impl_->width) * impl_->height);
    for (int y = 0; y < impl_->height; ++y)
    {
        const auto* row = sparkless.ptr<float>(camera * impl_->height + y);
        std::copy(row, row + impl_->width,
                  output.begin() + static_cast<std::size_t>(y) * impl_->width);
    }
}

Image read_pgm(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Could not open PGM: " + filename);
    std::string magic;
    std::getline(input, magic);
    if (magic != "P5")
        throw std::runtime_error("Expected P5 PGM: " + filename);
    std::string line;
    std::vector<std::string> tokens;
    while (tokens.size() < 3 && std::getline(input, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.resize(comment);
        std::string token;
        for (char character : line)
        {
            if (std::isspace(static_cast<unsigned char>(character)))
            {
                if (!token.empty())
                {
                    tokens.push_back(token);
                    token.clear();
                }
            }
            else
                token += character;
        }
        if (!token.empty())
            tokens.push_back(token);
    }
    if (tokens.size() < 3)
        throw std::runtime_error("Incomplete PGM header: " + filename);
    const int width = std::stoi(tokens[0]);
    const int height = std::stoi(tokens[1]);
    if (std::stoi(tokens[2]) != 65535)
        throw std::runtime_error("Expected 16-bit PGM: " + filename);
    Image result(width, height);
    for (int& pixel : result.data)
    {
        unsigned char bytes[2];
        if (!input.read(reinterpret_cast<char*>(bytes), 2))
            throw std::runtime_error("Incomplete PGM data: " + filename);
        pixel = (static_cast<int>(bytes[0]) << 8) | bytes[1];
    }
    return result;
}

void write_pgm(const std::string& filename, const Image& image)
{
    std::ofstream output(filename, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create PGM: " + filename);
    output << "P5\n" << image.width << ' ' << image.height << "\n65535\n";
    for (int value : image.data)
    {
        const uint16_t pixel = static_cast<uint16_t>(std::max(0, std::min(65535, value)));
        output.put(static_cast<char>(pixel >> 8));
        output.put(static_cast<char>(pixel & 0xff));
    }
}

void write_pgm(const std::string& filename, const uint16_t* pixels,
               int width, int height)
{
    std::ofstream output(filename, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create PGM: " + filename);
    output << "P5\n" << width << ' ' << height << "\n65535\n";
    const std::size_t count = static_cast<std::size_t>(width) * height;
    for (std::size_t index = 0; index < count; ++index)
    {
        const uint16_t pixel = pixels[index];
        output.put(static_cast<char>(pixel >> 8));
        output.put(static_cast<char>(pixel & 0xff));
    }
}
