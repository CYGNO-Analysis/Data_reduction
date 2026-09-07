#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<int> data;

    Image() = default;
    Image(int width, int height);
};

struct FloatImage
{
    int width = 0;
    int height = 0;
    std::vector<float> data;

    FloatImage() = default;
    FloatImage(int width, int height);
};

struct TriggerTiming
{
    std::size_t triggered_pixels = 0;
    double upload_ms = 0.0;
    double pedestal_ms = 0.0;
    double laplacian_ms = 0.0;
    double spark_threshold_ms = 0.0;
    double spark_dilation_ms = 0.0;
    double spark_mask_ms = 0.0;
    double gaussian_ms = 0.0;
    double centroid_threshold_ms = 0.0;
    double centroid_dilation_ms = 0.0;
    double final_mask_ms = 0.0;
    double download_ms = 0.0;
};

struct TriggerDebug
{
    FloatImage pedestal_subtracted;
    FloatImage laplacian;
    std::vector<uint8_t> spark_mask;
    std::vector<uint8_t> spark_dilated;
    FloatImage sparkless;
    FloatImage filtered;
    std::vector<uint8_t> centroid_mask;
    std::vector<uint8_t> centroid_dilated;
};

Image trigger_cuda(const Image& image, const FloatImage& pedestal,
                   int gaussian_kernel_size, float gaussian_sigma,
                   float spark_cut, float threshold_cut, int dilation_radius,
                   TriggerTiming* timing = nullptr);

    Image trigger_cuda_debug(const Image& image, const FloatImage& pedestal,
                     int gaussian_kernel_size, float gaussian_sigma,
                     float spark_cut, float threshold_cut, int dilation_radius,
                     TriggerDebug& debug);

Image read_pgm(const std::string& filename);
void write_pgm(const std::string& filename, const Image& image);
