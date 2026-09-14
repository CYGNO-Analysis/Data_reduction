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
    double mask_apply_ms = 0.0;
    double download_ms = 0.0;
    double cpu_output_conversion_ms = 0.0;
    // Measured directly via CUDA events (stage 0 start to stage 10 end), not a sum.
    double full_trigger_ms = 0.0;
};

class TriggerContext
{
public:
    TriggerContext(const FloatImage& pedestal, int gaussian_kernel_size,
                   float gaussian_sigma, float spark_cut,
                   float threshold_cut, int dilation_radius);
    ~TriggerContext();

    TriggerContext(const TriggerContext&) = delete;
    TriggerContext& operator=(const TriggerContext&) = delete;

    Image process_gpu_output(const Image& image, TriggerTiming* timing = nullptr);
    void process_gpu_output_pgm(const Image& image, const std::string& filename,
                                TriggerTiming* timing = nullptr);

private:
    Image process_impl(const Image& image, TriggerTiming* timing,
                       const std::string* output_filename = nullptr);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
