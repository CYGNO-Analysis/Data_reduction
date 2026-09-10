#include "trigger_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

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

namespace
{
using Clock = std::chrono::steady_clock;

void check_context_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
double context_elapsed_ms(T start, T end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void context_threshold(const float* source, std::size_t source_step,
                                  unsigned char* mask, std::size_t mask_step,
                                  float cut, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    const auto* source_row = reinterpret_cast<const float*>(
        reinterpret_cast<const unsigned char*>(source) + y * source_step);
    auto* mask_row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(mask) + y * mask_step);
    mask_row[x] = source_row[x] >= cut ? 255 : 0;
}

__global__ void context_zero_mask(float* image, std::size_t image_step,
                                  const unsigned char* mask, std::size_t mask_step,
                                  int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    const auto* mask_row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(mask) + y * mask_step);
    auto* image_row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(image) + y * image_step);
    if (mask_row[x] != 0)
        image_row[x] = 0.0f;
}

__global__ void context_apply_mask(const uint16_t* image, std::size_t image_step,
                                   const unsigned char* mask, std::size_t mask_step,
                                   uint16_t* output, std::size_t output_step,
                                   int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    const auto* image_row = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const unsigned char*>(image) + y * image_step);
    const auto* mask_row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(mask) + y * mask_step);
    auto* output_row = reinterpret_cast<uint16_t*>(
        reinterpret_cast<unsigned char*>(output) + y * output_step);
    output_row[x] = mask_row[x] != 0 ? image_row[x] : 0;
}

__global__ void context_count_mask(const unsigned char* mask, std::size_t mask_step,
                                   unsigned int* count, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(width) * height;
    if (index >= total)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    const auto* row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(mask) + y * mask_step);
    if (row[x] != 0)
        atomicAdd(count, 1U);
}

__constant__ float context_gaussian_weights[64];

__device__ float context_read_float(const float* data, std::size_t step,
                                    int width, int height, int y, int x)
{
    if (y < 0 || y >= height || x < 0 || x >= width)
        return 0.0f;
    const auto* row = reinterpret_cast<const float*>(
        reinterpret_cast<const unsigned char*>(data) + y * step);
    return row[x];
}

__global__ void context_gaussian_horizontal(const float* source, std::size_t source_step,
                                            float* output, std::size_t output_step,
                                            int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
        value += context_gaussian_weights[offset + radius]
            * context_read_float(source, source_step, width, height, y, x + offset);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + y * output_step);
    row[x] = value;
}

__global__ void context_gaussian_vertical(const float* source, std::size_t source_step,
                                          float* output, std::size_t output_step,
                                          int radius, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    float value = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset)
        value += context_gaussian_weights[offset + radius]
            * context_read_float(source, source_step, width, height, y + offset, x);
    auto* row = reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(output) + y * output_step);
    row[x] = value;
}

__device__ unsigned char context_read_mask(const unsigned char* data,
                                           std::size_t step, int width, int height,
                                           int y, int x)
{
    if (y < 0 || y >= height || x < 0 || x >= width)
        return 0;
    const auto* row = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const unsigned char*>(data) + y * step);
    return row[x];
}

__global__ void context_dilate_horizontal(const unsigned char* source,
                                          std::size_t source_step,
                                          unsigned char* output,
                                          std::size_t output_step, int radius,
                                          int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    unsigned char value = 0;
    for (int offset = -radius; offset <= radius; ++offset)
        value = max(value, context_read_mask(source, source_step, width, height,
                                              y, x + offset));
    auto* row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(output) + y * output_step);
    row[x] = value;
}

__global__ void context_dilate_vertical(const unsigned char* source,
                                        std::size_t source_step,
                                        unsigned char* output,
                                        std::size_t output_step, int radius,
                                        int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count)
        return;
    const int y = static_cast<int>(index / width);
    const int x = static_cast<int>(index % width);
    unsigned char value = 0;
    for (int offset = -radius; offset <= radius; ++offset)
        value = max(value, context_read_mask(source, source_step, width, height,
                                              y + offset, x));
    auto* row = reinterpret_cast<unsigned char*>(
        reinterpret_cast<unsigned char*>(output) + y * output_step);
    row[x] = value;
}
}

struct TriggerContext::Impl
{
    int width;
    int height;
    int kernel_size;
    int gaussian_radius;
    float sigma;
    float spark_cut;
    float threshold_cut;
    int dilation_radius;
    cv::cuda::GpuMat pedestal_gpu;
    cv::cuda::GpuMat image_gpu;
    cv::cuda::GpuMat result_gpu;
    unsigned int* count_gpu = nullptr;
    cv::cuda::GpuMat image_float;
    cv::cuda::GpuMat pedestal_subtracted;
    cv::cuda::GpuMat spark_mask;
    cv::cuda::GpuMat spark_dilated_horizontal;
    cv::cuda::GpuMat spark_dilated;
    cv::cuda::GpuMat centroid_mask;
    cv::cuda::GpuMat centroid_dilated_horizontal;
    cv::cuda::GpuMat centroid_dilated;
    cv::cuda::GpuMat laplacian_image;
    cv::cuda::GpuMat sparkless_image;
    cv::cuda::GpuMat gaussian_horizontal_image;
    cv::cuda::GpuMat filtered_image;
    cv::Ptr<cv::cuda::Filter> laplacian;
    uint16_t* input_host = nullptr;
        uint16_t* output_host = nullptr;
    cv::cuda::Stream stream;
    static constexpr int stage_count = 11;
    cudaEvent_t stage_start[stage_count]{};
    cudaEvent_t stage_end[stage_count]{};

    Impl(const FloatImage& pedestal, int gaussian_kernel_size, float gaussian_sigma,
         float spark, float threshold, int radius)
                : width(pedestal.width), height(pedestal.height), kernel_size(gaussian_kernel_size),
                    gaussian_radius(gaussian_kernel_size / 2),
          sigma(gaussian_sigma), spark_cut(spark), threshold_cut(threshold),
          dilation_radius(radius),
          pedestal_gpu(pedestal.height, pedestal.width, CV_32F),
          image_gpu(pedestal.height, pedestal.width, CV_16U),
          result_gpu(pedestal.height, pedestal.width, CV_16U),
          image_float(pedestal.height, pedestal.width, CV_32F),
          pedestal_subtracted(pedestal.height, pedestal.width, CV_32F),
          spark_mask(pedestal.height, pedestal.width, CV_8U),
          spark_dilated_horizontal(pedestal.height, pedestal.width, CV_8U),
          spark_dilated(pedestal.height, pedestal.width, CV_8U),
          centroid_mask(pedestal.height, pedestal.width, CV_8U),
          centroid_dilated_horizontal(pedestal.height, pedestal.width, CV_8U),
          centroid_dilated(pedestal.height, pedestal.width, CV_8U),
          laplacian_image(pedestal.height, pedestal.width, CV_32F),
          sparkless_image(pedestal.height, pedestal.width, CV_32F),
          gaussian_horizontal_image(pedestal.height, pedestal.width, CV_32F),
          filtered_image(pedestal.height, pedestal.width, CV_32F)
    {
        if (width <= 0 || height <= 0 || kernel_size <= 0 || kernel_size % 2 == 0
            || sigma <= 0.0f || radius < 0)
            throw std::runtime_error("Invalid persistent trigger parameters");
        check_context_cuda(cudaMallocHost(
            reinterpret_cast<void**>(&input_host),
            static_cast<std::size_t>(width) * height * sizeof(uint16_t)),
            "allocate pinned input");
        check_context_cuda(cudaMallocHost(
            reinterpret_cast<void**>(&output_host),
            static_cast<std::size_t>(width) * height * sizeof(uint16_t)),
            "allocate pinned output");
            check_context_cuda(cudaMalloc(reinterpret_cast<void**>(&count_gpu), sizeof(unsigned int)),
                       "allocate mask count");
        std::vector<float> weights(kernel_size);
        const float denominator = 2.0f * sigma * sigma;
        float weight_sum = 0.0f;
        for (int index = 0; index < kernel_size; ++index)
        {
            const int offset = index - gaussian_radius;
            weights[index] = std::exp(-(offset * offset) / denominator);
            weight_sum += weights[index];
        }
        for (float& weight : weights)
            weight /= weight_sum;
        check_context_cuda(cudaMemcpyToSymbol(context_gaussian_weights, weights.data(),
                                              weights.size() * sizeof(float)),
                           "copy Gaussian weights");
        for (int stage = 0; stage < stage_count; ++stage)
        {
            check_context_cuda(cudaEventCreate(&stage_start[stage]), "create stage start event");
            check_context_cuda(cudaEventCreate(&stage_end[stage]), "create stage end event");
        }

        cv::Mat pedestal_host(height, width, CV_32F,
                              const_cast<float*>(pedestal.data.data()));
        pedestal_gpu.upload(pedestal_host);
        const cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) <<
            -1.0f, -1.0f, -1.0f,
            -1.0f,  8.0f, -1.0f,
            -1.0f, -1.0f, -1.0f);
        laplacian = cv::cuda::createLinearFilter(
            CV_32F, CV_32F, laplacian_kernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);
    }

    ~Impl()
    {
        for (int stage = 0; stage < stage_count; ++stage)
        {
            if (stage_start[stage])
                cudaEventDestroy(stage_start[stage]);
            if (stage_end[stage])
                cudaEventDestroy(stage_end[stage]);
        }
        if (input_host)
            cudaFreeHost(input_host);
        if (output_host)
            cudaFreeHost(output_host);
        if (count_gpu)
            cudaFree(count_gpu);
    }
};

TriggerContext::TriggerContext(const FloatImage& pedestal, int gaussian_kernel_size,
                               float gaussian_sigma, float spark_cut,
                               float threshold_cut, int dilation_radius)
    : impl_(new Impl(pedestal, gaussian_kernel_size, gaussian_sigma, spark_cut,
                     threshold_cut, dilation_radius))
{
}

TriggerContext::~TriggerContext() = default;

Image TriggerContext::process_gpu_output(const Image& image, TriggerTiming* timing)
{
    return process_impl(image, timing);
}

void TriggerContext::process_gpu_output_pgm(const Image& image,
                                            const std::string& filename,
                                            TriggerTiming* timing)
{
    process_impl(image, timing, &filename);
}


Image TriggerContext::process_impl(const Image& image, TriggerTiming* timing,
                                   const std::string* output_filename)
{
    if (image.width != impl_->width || image.height != impl_->height)
        throw std::runtime_error("Image dimensions do not match persistent pedestal");

    const std::size_t count = image.data.size();
    std::transform(image.data.begin(), image.data.end(), impl_->input_host,
                   [](int pixel) { return static_cast<uint16_t>(pixel); });
    cv::Mat image_host(image.height, image.width, CV_16U, impl_->input_host);
    auto record_start = [&](int stage) {
        check_context_cuda(cudaEventRecord(impl_->stage_start[stage],
                                           reinterpret_cast<cudaStream_t>(
                                               impl_->stream.cudaPtr())),
                           "record stage start");
    };
    auto record_end = [&](int stage) {
        check_context_cuda(cudaEventRecord(impl_->stage_end[stage],
                                           reinterpret_cast<cudaStream_t>(
                                               impl_->stream.cudaPtr())),
                           "record stage end");
    };

    record_start(0);
    impl_->image_gpu.upload(image_host, impl_->stream);
    record_end(0);

    constexpr int block_size = 256;
    const int blocks = static_cast<int>((count + block_size - 1) / block_size);
    int width = image.width;
    int height = image.height;
    record_start(1);
    impl_->image_gpu.convertTo(impl_->image_float, CV_32F, impl_->stream);
    cv::cuda::subtract(impl_->image_float, impl_->pedestal_gpu,
                       impl_->pedestal_subtracted, cv::noArray(), -1, impl_->stream);
    impl_->pedestal_subtracted.copyTo(impl_->sparkless_image, impl_->stream);
    record_end(1);

    record_start(2);
    impl_->laplacian->apply(impl_->pedestal_subtracted, impl_->laplacian_image,
                            impl_->stream);
    record_end(2);

    float* laplacian_pointer = impl_->laplacian_image.ptr<float>();
    unsigned char* spark_pointer = impl_->spark_mask.ptr<unsigned char>();
    std::size_t laplacian_step = impl_->laplacian_image.step;
    std::size_t spark_step = impl_->spark_mask.step;
    void* threshold_arguments[] = {&laplacian_pointer, &laplacian_step,
        &spark_pointer, &spark_step, &impl_->spark_cut, &width, &height};
    record_start(3);
    check_context_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(context_threshold),
                                        dim3(blocks), dim3(block_size), threshold_arguments,
                                        0, reinterpret_cast<cudaStream_t>(
                                            impl_->stream.cudaPtr())),
                       "persistent spark threshold");
    record_end(3);

    record_start(4);
    unsigned char* spark_horizontal_pointer = impl_->spark_dilated_horizontal.ptr<unsigned char>();
    unsigned char* spark_dilated_pointer = impl_->spark_dilated.ptr<unsigned char>();
    std::size_t spark_horizontal_step = impl_->spark_dilated_horizontal.step;
    std::size_t spark_dilated_step = impl_->spark_dilated.step;
    int spark_radius = 1;
    void* spark_horizontal_arguments[] = {&spark_pointer, &spark_step,
        &spark_horizontal_pointer, &spark_horizontal_step, &spark_radius,
        &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_dilate_horizontal),
        dim3(blocks), dim3(block_size), spark_horizontal_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent spark horizontal dilation");
    void* spark_vertical_arguments[] = {&spark_horizontal_pointer,
        &spark_horizontal_step, &spark_dilated_pointer, &spark_dilated_step,
        &spark_radius, &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_dilate_vertical),
        dim3(blocks), dim3(block_size), spark_vertical_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent spark vertical dilation");
    record_end(4);

    float* sparkless_pointer = impl_->sparkless_image.ptr<float>();
    std::size_t sparkless_step = impl_->sparkless_image.step;
    void* zero_arguments[] = {&sparkless_pointer, &sparkless_step,
        &spark_dilated_pointer, &spark_dilated_step, &width, &height};
    record_start(5);
    check_context_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(context_zero_mask),
                                        dim3(blocks), dim3(block_size), zero_arguments,
                                        0, reinterpret_cast<cudaStream_t>(
                                            impl_->stream.cudaPtr())),
                       "persistent spark mask");
    record_end(5);

    record_start(6);
    float* gaussian_horizontal_pointer = impl_->gaussian_horizontal_image.ptr<float>();
    std::size_t gaussian_horizontal_step = impl_->gaussian_horizontal_image.step;
    float* filtered_pointer = impl_->filtered_image.ptr<float>();
    std::size_t filtered_step = impl_->filtered_image.step;
    void* gaussian_horizontal_arguments[] = {&sparkless_pointer, &sparkless_step,
        &gaussian_horizontal_pointer, &gaussian_horizontal_step, &impl_->gaussian_radius,
        &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_gaussian_horizontal),
        dim3(blocks), dim3(block_size), gaussian_horizontal_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent Gaussian horizontal");
    void* gaussian_vertical_arguments[] = {&gaussian_horizontal_pointer,
        &gaussian_horizontal_step, &filtered_pointer, &filtered_step,
        &impl_->gaussian_radius, &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_gaussian_vertical),
        dim3(blocks), dim3(block_size), gaussian_vertical_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent Gaussian vertical");
    record_end(6);

    unsigned char* centroid_pointer = impl_->centroid_mask.ptr<unsigned char>();
    std::size_t centroid_step = impl_->centroid_mask.step;
    void* centroid_arguments[] = {&filtered_pointer, &filtered_step,
        &centroid_pointer, &centroid_step, &impl_->threshold_cut, &width, &height};
    record_start(7);
    check_context_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(context_threshold),
                                        dim3(blocks), dim3(block_size), centroid_arguments,
                                        0, reinterpret_cast<cudaStream_t>(
                                            impl_->stream.cudaPtr())),
                       "persistent centroid threshold");
    record_end(7);

    record_start(8);
    unsigned char* centroid_horizontal_pointer =
        impl_->centroid_dilated_horizontal.ptr<unsigned char>();
    unsigned char* centroid_dilated_pointer = impl_->centroid_dilated.ptr<unsigned char>();
    std::size_t centroid_horizontal_step = impl_->centroid_dilated_horizontal.step;
    std::size_t centroid_dilated_step = impl_->centroid_dilated.step;
    void* centroid_horizontal_arguments[] = {&centroid_pointer, &centroid_step,
        &centroid_horizontal_pointer, &centroid_horizontal_step, &impl_->dilation_radius,
        &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_dilate_horizontal),
        dim3(blocks), dim3(block_size), centroid_horizontal_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent centroid horizontal dilation");
    void* centroid_vertical_arguments[] = {&centroid_horizontal_pointer,
        &centroid_horizontal_step, &centroid_dilated_pointer, &centroid_dilated_step,
        &impl_->dilation_radius, &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_dilate_vertical),
        dim3(blocks), dim3(block_size), centroid_vertical_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent centroid vertical dilation");
    record_end(8);

    record_start(9);
    uint16_t* image_pointer = impl_->image_gpu.ptr<uint16_t>();
    uint16_t* result_pointer = impl_->result_gpu.ptr<uint16_t>();
    std::size_t image_step = impl_->image_gpu.step;
    std::size_t result_step = impl_->result_gpu.step;
    void* output_arguments[] = {&image_pointer, &image_step,
        &centroid_dilated_pointer, &centroid_dilated_step, &result_pointer,
        &result_step, &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_apply_mask),
        dim3(blocks), dim3(block_size), output_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "persistent GPU output mask");
    check_context_cuda(cudaMemsetAsync(impl_->count_gpu, 0, sizeof(unsigned int),
                                       reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
                       "clear mask count");
    void* count_arguments[] = {&centroid_dilated_pointer, &centroid_dilated_step,
        &impl_->count_gpu, &width, &height};
    check_context_cuda(cudaLaunchKernel(
        reinterpret_cast<const void*>(context_count_mask),
        dim3(blocks), dim3(block_size), count_arguments, 0,
        reinterpret_cast<cudaStream_t>(impl_->stream.cudaPtr())),
        "count mask pixels");
    record_end(9);
    record_start(10);
    cv::Mat result_gpu_host(height, width, CV_16U, impl_->output_host);
    impl_->result_gpu.download(result_gpu_host, impl_->stream);
    record_end(10);
    impl_->stream.waitForCompletion();
    auto elapsed_stage = [&](int stage) {
        float milliseconds = 0.0f;
        check_context_cuda(cudaEventElapsedTime(&milliseconds,
                                                impl_->stage_start[stage],
                                                impl_->stage_end[stage]),
                           "measure CUDA stage");
        return static_cast<double>(milliseconds);
    };
    if (timing)
    {
        timing->upload_ms = elapsed_stage(0);
        timing->pedestal_ms = elapsed_stage(1);
        timing->laplacian_ms = elapsed_stage(2);
        timing->spark_threshold_ms = elapsed_stage(3);
        timing->spark_dilation_ms = elapsed_stage(4);
        timing->spark_mask_ms = elapsed_stage(5);
        timing->gaussian_ms = elapsed_stage(6);
        timing->centroid_threshold_ms = elapsed_stage(7);
        timing->centroid_dilation_ms = elapsed_stage(8);
        timing->mask_apply_ms = elapsed_stage(9);
        timing->download_ms = elapsed_stage(10);
        float full_elapsed_ms = 0.0f;
        check_context_cuda(cudaEventElapsedTime(&full_elapsed_ms, impl_->stage_start[0],
                                                impl_->stage_end[10]),
                           "measure full trigger span");
        timing->full_trigger_ms = static_cast<double>(full_elapsed_ms);
    }
    Image result(image.width, image.height);
    cv::Mat result_host(image.height, image.width, CV_32S, result.data.data());
    if (output_filename)
    {
        std::ofstream output(*output_filename, std::ios::binary);
        if (!output)
            throw std::runtime_error("Could not create PGM: " + *output_filename);
        output << "P5\n" << width << ' ' << height << "\n65535\n";
        for (std::size_t index = 0; index < count; ++index)
        {
            const uint16_t pixel = impl_->output_host[index];
            output.put(static_cast<char>(pixel >> 8));
            output.put(static_cast<char>(pixel & 0xff));
        }
        if (timing)
            timing->cpu_output_conversion_ms = 0.0;
    }
    else
    {
        const auto conversion_start = Clock::now();
        result_gpu_host.convertTo(result_host, CV_32S);
        if (timing)
            timing->cpu_output_conversion_ms = context_elapsed_ms(
                conversion_start, Clock::now());
    }
    if (timing)
    {
        unsigned int count_result = 0;
        check_context_cuda(cudaMemcpy(&count_result, impl_->count_gpu, sizeof(count_result),
                                      cudaMemcpyDeviceToHost),
                           "download mask count");
        timing->triggered_pixels = count_result;
    }
    return result;
}
