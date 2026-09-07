#pragma once

#include <cstdint>
#include <memory>
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
    double cpu_output_conversion_ms = 0.0;
};

Image trigger_cuda(const Image& image, const FloatImage& pedestal,
                   int gaussian_kernel_size, float gaussian_sigma,
                   float spark_cut, float threshold_cut, int dilation_radius,
                   TriggerTiming* timing = nullptr);

class TriggerContext
{
public:
    TriggerContext(const FloatImage& pedestal, int gaussian_kernel_size,
                   float gaussian_sigma, float spark_cut,
                   float threshold_cut, int dilation_radius);
    ~TriggerContext();

    TriggerContext(const TriggerContext&) = delete;
    TriggerContext& operator=(const TriggerContext&) = delete;

    Image process(const Image& image, TriggerTiming* timing = nullptr);
    Image process_gpu_output(const Image& image, TriggerTiming* timing = nullptr);
    void process_gpu_output_pgm(const Image& image, const std::string& filename,
                                TriggerTiming* timing = nullptr);
    void copy_centroid_mask(std::vector<uint8_t>& output);
    void copy_filtered_image(std::vector<float>& output);
    void copy_sparkless_image(std::vector<float>& output);

private:
    Image process_impl(const Image& image, TriggerTiming* timing, bool gpu_output,
                       const std::string* output_filename = nullptr);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void write_pgm(const std::string& filename, const Image& image);
