#include "trigger_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
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

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
double elapsed_ms(T start, T end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void threshold_kernel(const float* source, std::size_t source_step,
                                unsigned char* mask, std::size_t mask_step,
                                float cut, int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
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

__global__ void zero_mask_kernel(float* image, std::size_t image_step,
                                 const unsigned char* mask, std::size_t mask_step,
                                 int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
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
}

Image trigger_cuda(const Image& image, const FloatImage& pedestal,
                   int gaussian_kernel_size, float gaussian_sigma,
                   float spark_cut, float threshold_cut, int dilation_radius,
                   TriggerTiming* timing)
{
    if (image.width != pedestal.width || image.height != pedestal.height)
        throw std::runtime_error("Image and pedestal must have the same dimensions.");
    if (gaussian_kernel_size <= 0 || gaussian_kernel_size % 2 == 0
        || gaussian_sigma <= 0.0f || dilation_radius < 0)
        throw std::runtime_error("Invalid trigger parameters");

    const std::size_t count = image.data.size();
    const auto upload_start = Clock::now();
    std::vector<uint16_t> image_16(count);
    std::transform(image.data.begin(), image.data.end(), image_16.begin(),
                   [](int pixel) { return static_cast<uint16_t>(pixel); });
    cv::Mat image_host(image.height, image.width, CV_16U, image_16.data());
    cv::Mat pedestal_host(pedestal.height, pedestal.width, CV_32F,
                          const_cast<float*>(pedestal.data.data()));
    const auto processing_start = Clock::now();

    cv::cuda::GpuMat image_gpu(image_host);
    cv::cuda::GpuMat pedestal_gpu(pedestal_host);
    cv::cuda::GpuMat image_float_gpu(image.height, image.width, CV_32F);
    cv::cuda::GpuMat image_pedsub(image.height, image.width, CV_32F);
    cv::cuda::GpuMat mask_spark(image.height, image.width, CV_8U);
    cv::cuda::GpuMat mask_spark_dilated(image.height, image.width, CV_8U);
    cv::cuda::GpuMat mask_centroids(image.height, image.width, CV_8U);
    cv::cuda::GpuMat mask_centroids_dilated(image.height, image.width, CV_8U);
    cv::cuda::GpuMat image_laplacian(image.height, image.width, CV_32F);
    cv::cuda::GpuMat image_sparkless(image.height, image.width, CV_32F);
    cv::cuda::GpuMat image_filtered(image.height, image.width, CV_32F);
    if (timing)
        timing->upload_ms = elapsed_ms(upload_start, Clock::now());

    auto stage_start = Clock::now();
    image_gpu.convertTo(image_float_gpu, CV_32F);
    cv::cuda::subtract(image_float_gpu, pedestal_gpu, image_pedsub);
    image_pedsub.copyTo(image_sparkless);
    check_cuda(cudaDeviceSynchronize(), "pedestal synchronize");
    if (timing)
        timing->pedestal_ms = elapsed_ms(stage_start, Clock::now());

    constexpr int block_size = 256;
    const int blocks = static_cast<int>((count + block_size - 1) / block_size);
    int width = image.width;
    int height = image.height;
    const cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) <<
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f);
    auto laplacian = cv::cuda::createLinearFilter(
        CV_32F, CV_32F, laplacian_kernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);
    stage_start = Clock::now();
    laplacian->apply(image_pedsub, image_laplacian);
    check_cuda(cudaDeviceSynchronize(), "Laplacian synchronize");
    if (timing)
        timing->laplacian_ms = elapsed_ms(stage_start, Clock::now());

    float* laplacian_pointer = image_laplacian.ptr<float>();
    unsigned char* spark_pointer = mask_spark.ptr<unsigned char>();
    std::size_t laplacian_step = image_laplacian.step;
    std::size_t spark_step = mask_spark.step;
    void* threshold_arguments[] = {
        &laplacian_pointer, &laplacian_step, &spark_pointer, &spark_step,
        &spark_cut, &width, &height
    };
    stage_start = Clock::now();
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(threshold_kernel),
                                dim3(blocks), dim3(block_size), threshold_arguments,
                                0, nullptr), "spark threshold");
    check_cuda(cudaDeviceSynchronize(), "spark threshold synchronize");
    if (timing)
        timing->spark_threshold_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    auto spark_dilate = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8U, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1));
    spark_dilate->apply(mask_spark, mask_spark_dilated);
    check_cuda(cudaDeviceSynchronize(), "spark dilation synchronize");
    if (timing)
        timing->spark_dilation_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    float* sparkless_pointer = image_sparkless.ptr<float>();
    unsigned char* dilated_spark_pointer = mask_spark_dilated.ptr<unsigned char>();
    std::size_t sparkless_step = image_sparkless.step;
    std::size_t dilated_spark_step = mask_spark_dilated.step;
    void* zero_arguments[] = {
        &sparkless_pointer, &sparkless_step, &dilated_spark_pointer,
        &dilated_spark_step, &width, &height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(zero_mask_kernel),
                                dim3(blocks), dim3(block_size), zero_arguments,
                                0, nullptr), "spark mask");
    check_cuda(cudaDeviceSynchronize(), "spark mask synchronize");
    if (timing)
        timing->spark_mask_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    auto gaussian = cv::cuda::createGaussianFilter(
        CV_32F, CV_32F, cv::Size(gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma, gaussian_sigma, cv::BORDER_CONSTANT);
    gaussian->apply(image_sparkless, image_filtered);
    check_cuda(cudaDeviceSynchronize(), "Gaussian synchronize");
    if (timing)
        timing->gaussian_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    float* filtered_pointer = image_filtered.ptr<float>();
    unsigned char* centroid_pointer = mask_centroids.ptr<unsigned char>();
    std::size_t filtered_step = image_filtered.step;
    std::size_t centroid_step = mask_centroids.step;
    void* centroid_arguments[] = {
        &filtered_pointer, &filtered_step, &centroid_pointer, &centroid_step,
        &threshold_cut, &width, &height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(threshold_kernel),
                                dim3(blocks), dim3(block_size), centroid_arguments,
                                0, nullptr), "centroid threshold");
    check_cuda(cudaDeviceSynchronize(), "centroid threshold synchronize");
    if (timing)
        timing->centroid_threshold_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    auto centroid_dilate = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8U,
        cv::Mat::ones(2 * dilation_radius + 1, 2 * dilation_radius + 1, CV_8U),
        cv::Point(-1, -1));
    centroid_dilate->apply(mask_centroids, mask_centroids_dilated);
    check_cuda(cudaDeviceSynchronize(), "centroid dilation synchronize");
    if (timing)
        timing->centroid_dilation_ms = elapsed_ms(stage_start, Clock::now());

    const auto download_start = Clock::now();
    cv::Mat final_mask_host;
    mask_centroids_dilated.download(final_mask_host);
    if (timing)
        timing->download_ms = elapsed_ms(download_start, Clock::now());
    Image result(image.width, image.height);
    cv::Mat result_mat(image.height, image.width, CV_32S, result.data.data());
    cv::Mat original_host(image.height, image.width, CV_32S,
                         const_cast<int*>(image.data.data()));
    original_host.copyTo(result_mat);
    cv::Mat inverse_mask;
    cv::compare(final_mask_host, 0, inverse_mask, cv::CMP_EQ);
    result_mat.setTo(0, inverse_mask);

    if (timing)
    {
        timing->triggered_pixels = static_cast<std::size_t>(
            cv::countNonZero(final_mask_host));
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
