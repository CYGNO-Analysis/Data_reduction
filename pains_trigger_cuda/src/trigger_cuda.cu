#include "trigger_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

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
    if (index < count)
    {
        const int y = static_cast<int>(index / width);
        const int x = static_cast<int>(index % width);
        const auto* source_row = reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(source) + y * source_step);
        auto* mask_row = reinterpret_cast<unsigned char*>(
            reinterpret_cast<unsigned char*>(mask) + y * mask_step);
        mask_row[x] = source_row[x] >= cut ? 255 : 0;
    }
}

__global__ void zero_mask_kernel(float* image, std::size_t image_step,
                                 const unsigned char* mask, std::size_t mask_step,
                                 int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index < count)
    {
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

__global__ void apply_mask_kernel(const int* image, std::size_t image_step,
                                  const unsigned char* mask, std::size_t mask_step,
                                  int* result, std::size_t result_step,
                                  int width, int height)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index < count)
    {
        const int y = static_cast<int>(index / width);
        const int x = static_cast<int>(index % width);
        const auto* image_row = reinterpret_cast<const int*>(
            reinterpret_cast<const unsigned char*>(image) + y * image_step);
        const auto* mask_row = reinterpret_cast<const unsigned char*>(
            reinterpret_cast<const unsigned char*>(mask) + y * mask_step);
        auto* result_row = reinterpret_cast<int*>(
            reinterpret_cast<unsigned char*>(result) + y * result_step);
        result_row[x] = mask_row[x] != 0 ? image_row[x] : 0;
    }
}

cv::Mat read_pgm_host(const std::string& filename)
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
    const int maximum = std::stoi(tokens[2]);
    if (maximum != 65535)
        throw std::runtime_error("Expected 16-bit PGM: " + filename);

    cv::Mat result(height, width, CV_32S);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            unsigned char bytes[2];
            if (!input.read(reinterpret_cast<char*>(bytes), 2))
                throw std::runtime_error("Incomplete PGM data: " + filename);
            result.at<int>(y, x) = (static_cast<int>(bytes[0]) << 8) | bytes[1];
        }
    return result;
}
}

Image trigger_cuda_impl(const Image& image, const FloatImage& pedestal,
                        int gaussian_kernel_size, float gaussian_sigma,
                        float spark_cut, float threshold_cut, int dilation_radius,
                        TriggerTiming* timing, TriggerDebug* debug)
{
    if (image.width != pedestal.width || image.height != pedestal.height)
        throw std::runtime_error("Image and pedestal must have the same dimensions.");
    if (gaussian_kernel_size <= 0 || gaussian_kernel_size % 2 == 0)
        throw std::runtime_error("Gaussian kernel size must be positive and odd.");
    if (gaussian_sigma <= 0.0f || dilation_radius < 0)
        throw std::runtime_error("Invalid Gaussian sigma or dilation radius.");

    const std::size_t count = image.data.size();
    const auto upload_start = Clock::now();
    cv::Mat image_host(image.height, image.width, CV_32S,
                       const_cast<int*>(image.data.data()));
    cv::Mat pedestal_host(pedestal.height, pedestal.width, CV_32F,
                          const_cast<float*>(pedestal.data.data()));
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
    cv::cuda::GpuMat result_gpu(image.height, image.width, CV_32S);
    if (timing)
        timing->upload_ms = elapsed_ms(upload_start, Clock::now());

    auto kernel_start = Clock::now();
    image_gpu.convertTo(image_float_gpu, CV_32F);
    cv::cuda::subtract(image_float_gpu, pedestal_gpu, image_pedsub);
    image_pedsub.copyTo(image_sparkless);
    if (timing)
        timing->pedestal_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    constexpr int block_size = 256;
    const int blocks = static_cast<int>((count + block_size - 1) / block_size);
    const cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) <<
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f);
    auto laplacian = cv::cuda::createLinearFilter(
        CV_32F, CV_32F, laplacian_kernel, cv::Point(-1, -1), cv::BORDER_CONSTANT);
    laplacian->apply(image_pedsub, image_laplacian);
    if (timing)
        timing->laplacian_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    float* laplacian_pointer = image_laplacian.ptr<float>();
    unsigned char* spark_pointer = mask_spark.ptr<unsigned char>();
    const std::size_t laplacian_step = image_laplacian.step;
    const std::size_t spark_step = mask_spark.step;
    const void* threshold_arguments[] = {
        &laplacian_pointer, &laplacian_step, &spark_pointer, &spark_step,
        &spark_cut, &image.width, &image.height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(threshold_kernel),
                                dim3(blocks), dim3(block_size),
                                const_cast<void**>(threshold_arguments), 0, nullptr),
               "spark threshold");
    check_cuda(cudaGetLastError(), "spark threshold");
    check_cuda(cudaDeviceSynchronize(), "spark threshold synchronize");
    if (timing)
        timing->spark_threshold_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    auto spark_dilate = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8U,
        cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1));
    spark_dilate->apply(mask_spark, mask_spark_dilated);
    if (timing)
        timing->spark_dilation_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    float* sparkless_pointer = image_sparkless.ptr<float>();
    unsigned char* dilated_spark_pointer = mask_spark_dilated.ptr<unsigned char>();
    const std::size_t sparkless_step = image_sparkless.step;
    const std::size_t dilated_spark_step = mask_spark_dilated.step;
    const void* zero_arguments[] = {
        &sparkless_pointer, &sparkless_step, &dilated_spark_pointer,
        &dilated_spark_step, &image.width, &image.height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(zero_mask_kernel),
                                dim3(blocks), dim3(block_size),
                                const_cast<void**>(zero_arguments), 0, nullptr),
               "spark mask");
    check_cuda(cudaGetLastError(), "spark mask");
    check_cuda(cudaDeviceSynchronize(), "spark mask synchronize");
    if (timing)
        timing->spark_mask_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    auto gaussian = cv::cuda::createGaussianFilter(
        CV_32F, CV_32F, cv::Size(gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma, gaussian_sigma, cv::BORDER_CONSTANT);
    gaussian->apply(image_sparkless, image_filtered);
    if (timing)
        timing->gaussian_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    float* filtered_pointer = image_filtered.ptr<float>();
    unsigned char* centroid_pointer = mask_centroids.ptr<unsigned char>();
    const std::size_t filtered_step = image_filtered.step;
    const std::size_t centroid_step = mask_centroids.step;
    const void* centroid_arguments[] = {
        &filtered_pointer, &filtered_step, &centroid_pointer, &centroid_step,
        &threshold_cut, &image.width, &image.height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(threshold_kernel),
                                dim3(blocks), dim3(block_size),
                                const_cast<void**>(centroid_arguments), 0, nullptr),
               "centroid threshold");
    check_cuda(cudaGetLastError(), "centroid threshold");
    check_cuda(cudaDeviceSynchronize(), "centroid threshold synchronize");
    if (timing)
        timing->centroid_threshold_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    auto centroid_dilate = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8U,
        cv::Mat::ones(2 * dilation_radius + 1, 2 * dilation_radius + 1, CV_8U),
        cv::Point(-1, -1));
    centroid_dilate->apply(mask_centroids, mask_centroids_dilated);
    if (timing)
        timing->centroid_dilation_ms = elapsed_ms(kernel_start, Clock::now());

    kernel_start = Clock::now();
    int* image_pointer = image_gpu.ptr<int>();
    const std::size_t image_step = image_gpu.step;
    int* result_pointer = result_gpu.ptr<int>();
    unsigned char* final_mask_pointer = mask_centroids_dilated.ptr<unsigned char>();
    const std::size_t result_step = result_gpu.step;
    const std::size_t final_mask_step = mask_centroids_dilated.step;
    const void* final_arguments[] = {
        &image_pointer, &image_step, &final_mask_pointer, &final_mask_step,
        &result_pointer, &result_step, &image.width, &image.height
    };
    check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(apply_mask_kernel),
                                dim3(blocks), dim3(block_size),
                                const_cast<void**>(final_arguments), 0, nullptr),
               "final mask");
    check_cuda(cudaGetLastError(), "final mask");
    check_cuda(cudaDeviceSynchronize(), "final mask synchronize");
    if (timing)
        timing->final_mask_ms = elapsed_ms(kernel_start, Clock::now());

    const auto download_start = Clock::now();
    cv::Mat result_host;
    result_gpu.download(result_host);
    cv::Mat final_mask_host;
    mask_centroids_dilated.download(final_mask_host);
    Image result(image.width, image.height);
    std::copy(reinterpret_cast<const int*>(result_host.data),
              reinterpret_cast<const int*>(result_host.data) + count,
              result.data.begin());
    if (timing)
    {
        timing->download_ms = elapsed_ms(download_start, Clock::now());
        timing->triggered_pixels = static_cast<std::size_t>(
            cv::countNonZero(final_mask_host)
        );
    }

    if (debug)
    {
        cv::Mat pedestal_subtracted_host;
        cv::Mat laplacian_host;
        cv::Mat spark_mask_host;
        cv::Mat spark_dilated_host;
        cv::Mat sparkless_host;
        cv::Mat filtered_host;
        cv::Mat centroid_mask_host;
        cv::Mat centroid_dilated_host;

        image_pedsub.download(pedestal_subtracted_host);
        image_laplacian.download(laplacian_host);
        mask_spark.download(spark_mask_host);
        mask_spark_dilated.download(spark_dilated_host);
        image_sparkless.download(sparkless_host);
        image_filtered.download(filtered_host);
        mask_centroids.download(centroid_mask_host);
        mask_centroids_dilated.download(centroid_dilated_host);

        auto copy_float = [&](const cv::Mat& source, FloatImage& target) {
            target = FloatImage(image.width, image.height);
            std::copy(reinterpret_cast<const float*>(source.data),
                      reinterpret_cast<const float*>(source.data) + count,
                      target.data.begin());
        };
        auto copy_mask = [&](const cv::Mat& source, std::vector<uint8_t>& target) {
            target.assign(source.data, source.data + count);
        };
        copy_float(pedestal_subtracted_host, debug->pedestal_subtracted);
        copy_float(laplacian_host, debug->laplacian);
        copy_mask(spark_mask_host, debug->spark_mask);
        copy_mask(spark_dilated_host, debug->spark_dilated);
        copy_float(sparkless_host, debug->sparkless);
        copy_float(filtered_host, debug->filtered);
        copy_mask(centroid_mask_host, debug->centroid_mask);
        copy_mask(centroid_dilated_host, debug->centroid_dilated);
    }
    return result;
}

Image trigger_cuda(const Image& image, const FloatImage& pedestal,
                   int gaussian_kernel_size, float gaussian_sigma,
                   float spark_cut, float threshold_cut, int dilation_radius,
                   TriggerTiming* timing)
{
    return trigger_cuda_impl(image, pedestal, gaussian_kernel_size,
                             gaussian_sigma, spark_cut, threshold_cut,
                             dilation_radius, timing, nullptr);
}

Image trigger_cuda_debug(const Image& image, const FloatImage& pedestal,
                         int gaussian_kernel_size, float gaussian_sigma,
                         float spark_cut, float threshold_cut, int dilation_radius,
                         TriggerDebug& debug)
{
    return trigger_cuda_impl(image, pedestal, gaussian_kernel_size,
                             gaussian_sigma, spark_cut, threshold_cut,
                             dilation_radius, nullptr, &debug);
}

Image read_pgm(const std::string& filename)
{
    const cv::Mat matrix = read_pgm_host(filename);
    Image result(matrix.cols, matrix.rows);
    std::copy(reinterpret_cast<const int*>(matrix.data),
              reinterpret_cast<const int*>(matrix.data) + result.data.size(),
              result.data.begin());
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
