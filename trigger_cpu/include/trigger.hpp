#pragma once

#include <cstdint>
#include <vector>

struct Image
{
    int width;
    int height;
    std::vector<int> data;

    Image() = default;
    Image(int width, int height);
};

struct FloatImage
{
    int width;
    int height;
    std::vector<float> data;

    FloatImage() = default;
    FloatImage(int width, int height);
};

struct Mask
{
    int width;
    int height;
    std::vector<uint8_t> data;

    Mask() = default;
    Mask(int width, int height);
};

struct TriggerTiming
{
    std::size_t triggered_pixels = 0;
    double pedestal_ms = 0.0;
    double spark_laplacian_ms = 0.0;
    double spark_threshold_ms = 0.0;
    double spark_dilation_ms = 0.0;
    double spark_mask_ms = 0.0;
    double gaussian_ms = 0.0;
    double centroid_threshold_ms = 0.0;
    double centroid_dilation_ms = 0.0;
    double final_mask_ms = 0.0;
};

FloatImage subtract_pedestal(const Image& image, const FloatImage& pedestal);
FloatImage laplacian_filter(const FloatImage& image);
Mask threshold(const FloatImage& image, float cut);
Mask dilate(const Mask& mask, int radius);
FloatImage apply_mask_zero(const FloatImage& image, const Mask& mask);
FloatImage gaussian_filter(const FloatImage& image, int kernel_size, float sigma);
FloatImage spark_correction_zero(const FloatImage& image, float cut);
Image apply_mask(const Image& image, const Mask& mask);

Image trigger(const Image& image, const FloatImage& pedestal,
              int gaussian_kernel_size, float gaussian_sigma, float spark_cut,
              float threshold_cut, int dilation_radius,
              TriggerTiming* timing = nullptr);