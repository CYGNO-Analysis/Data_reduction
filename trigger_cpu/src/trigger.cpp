#include "trigger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

Image::Image(int width, int height)
    : width(width), height(height), data(static_cast<std::size_t>(width) * height)
{
}

FloatImage::FloatImage(int width, int height)
    : width(width), height(height), data(static_cast<std::size_t>(width) * height)
{
}

Mask::Mask(int width, int height)
    : width(width), height(height), data(static_cast<std::size_t>(width) * height)
{
}

namespace
{
cv::Mat mask_to_cv(const Mask& mask)
{
    return cv::Mat(mask.height, mask.width, CV_8U,
                   const_cast<uint8_t*>(mask.data.data()));
}

FloatImage copy_float_mat(const cv::Mat& matrix)
{
    FloatImage result(matrix.cols, matrix.rows);
    std::copy(reinterpret_cast<const float*>(matrix.data),
              reinterpret_cast<const float*>(matrix.data) + result.data.size(),
              result.data.begin());
    return result;
}
}

FloatImage subtract_pedestal(const Image& image, const FloatImage& pedestal)
{
    if (image.width != pedestal.width || image.height != pedestal.height)
        throw std::runtime_error("Image and pedestal must have the same dimensions.");
    if (image.data.size() != pedestal.data.size())
        throw std::runtime_error("Image and pedestal data must have the same size.");

    cv::Mat image_mat(image.height, image.width, CV_32S,
                      const_cast<int*>(image.data.data()));
    cv::Mat pedestal_mat(pedestal.height, pedestal.width, CV_32F,
                         const_cast<float*>(pedestal.data.data()));
    cv::Mat result_mat;
    cv::subtract(image_mat, pedestal_mat, result_mat, cv::noArray(), CV_32F);
    return copy_float_mat(result_mat);
}

FloatImage laplacian_filter(const FloatImage& image)
{
    cv::Mat source(image.height, image.width, CV_32F,
                   const_cast<float*>(image.data.data()));
    const cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    );
    cv::Mat filtered;
    cv::filter2D(source, filtered, CV_32F, kernel, cv::Point(-1, -1),
                 0.0, cv::BORDER_CONSTANT);
    return copy_float_mat(filtered);
}

Mask threshold(const FloatImage& image, float cut)
{
    cv::Mat source(image.height, image.width, CV_32F,
                   const_cast<float*>(image.data.data()));
    cv::Mat compared;
    cv::compare(source, cut, compared, cv::CMP_GE);

    Mask result(image.width, image.height);
    std::copy(compared.data, compared.data + result.data.size(), result.data.begin());
    return result;
}

Mask dilate(const Mask& mask, int radius)
{
    if (radius < 0)
        throw std::runtime_error("Dilation radius must be non-negative.");

    cv::Mat source = mask_to_cv(mask);
    const cv::Mat kernel = cv::Mat::ones(2 * radius + 1,
                                         2 * radius + 1, CV_8U);
    cv::Mat dilated;
    cv::dilate(source, dilated, kernel, cv::Point(-1, -1), 1,
               cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

    Mask result(mask.width, mask.height);
    std::copy(dilated.data, dilated.data + result.data.size(), result.data.begin());
    return result;
}

FloatImage apply_mask_zero(const FloatImage& image, const Mask& mask)
{
    if (image.width != mask.width || image.height != mask.height)
        throw std::runtime_error("Image and mask must have the same dimensions.");

    cv::Mat source(image.height, image.width, CV_32F,
                   const_cast<float*>(image.data.data()));
    cv::Mat result_mat = source.clone();
    result_mat.setTo(0.0f, mask_to_cv(mask));
    return copy_float_mat(result_mat);
}

FloatImage gaussian_filter(const FloatImage& image, int kernel_size, float sigma)
{
    if (kernel_size <= 0 || kernel_size % 2 == 0)
        throw std::runtime_error("Gaussian kernel size must be positive and odd.");
    if (sigma <= 0.0f)
        throw std::runtime_error("Gaussian sigma must be positive.");

    cv::Mat source(image.height, image.width, CV_32F,
                   const_cast<float*>(image.data.data()));
    cv::Mat filtered;
    cv::GaussianBlur(source, filtered, cv::Size(kernel_size, kernel_size),
                     sigma, sigma, cv::BORDER_CONSTANT);
    return copy_float_mat(filtered);
}

FloatImage spark_correction_zero(const FloatImage& image, float cut)
{
    const FloatImage laplacian = laplacian_filter(image);
    const Mask mask = threshold(laplacian, cut);
    const Mask dilated_mask = dilate(mask, 1);
    return apply_mask_zero(image, dilated_mask);
}

Image apply_mask(const Image& image, const Mask& mask)
{
    if (image.width != mask.width || image.height != mask.height)
        throw std::runtime_error("Image and mask must have the same dimensions.");

    cv::Mat source(image.height, image.width, CV_32S,
                   const_cast<int*>(image.data.data()));
    cv::Mat result_mat = cv::Mat::zeros(image.height, image.width, CV_32S);
    source.copyTo(result_mat, mask_to_cv(mask));

    Image result(image.width, image.height);
    std::copy(reinterpret_cast<const int*>(result_mat.data),
              reinterpret_cast<const int*>(result_mat.data) + result.data.size(),
              result.data.begin());
    return result;
}

Image trigger(const Image& image, const FloatImage& pedestal,
              int gaussian_kernel_size, float gaussian_sigma,
              float spark_cut, float threshold_cut, int dilation_radius,
              TriggerTiming* timing)
{
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    auto stage_start = Clock::now();
    const FloatImage image_pedsub = subtract_pedestal(image, pedestal);
    auto stage_end = Clock::now();
    if (timing)
        timing->pedestal_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const FloatImage laplacian = laplacian_filter(image_pedsub);
    stage_end = Clock::now();
    if (timing)
        timing->spark_laplacian_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const Mask spark_mask = threshold(laplacian, spark_cut);
    stage_end = Clock::now();
    if (timing)
        timing->spark_threshold_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const Mask dilated_spark_mask = dilate(spark_mask, 1);
    stage_end = Clock::now();
    if (timing)
        timing->spark_dilation_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const FloatImage image_sparkless = apply_mask_zero(image_pedsub, dilated_spark_mask);
    stage_end = Clock::now();
    if (timing)
        timing->spark_mask_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const FloatImage image_filtered = gaussian_filter(
        image_sparkless, gaussian_kernel_size, gaussian_sigma
    );
    stage_end = Clock::now();
    if (timing)
        timing->gaussian_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const Mask centroids = threshold(image_filtered, threshold_cut);
    stage_end = Clock::now();
    if (timing)
        timing->centroid_threshold_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    const Mask mask = dilate(centroids, dilation_radius);
    stage_end = Clock::now();
    if (timing)
        timing->centroid_dilation_ms = elapsed_ms(stage_start, stage_end);

    stage_start = Clock::now();
    Image result = apply_mask(image, mask);
    stage_end = Clock::now();
    if (timing)
    {
        timing->final_mask_ms = elapsed_ms(stage_start, stage_end);
        timing->triggered_pixels = static_cast<std::size_t>(
            std::count_if(mask.data.begin(), mask.data.end(),
                          [](uint8_t value) { return value != 0; })
        );
    }

    return result;
}
