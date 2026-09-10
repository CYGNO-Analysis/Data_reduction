#pragma once

#include <array>
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

struct BatchTiming
{
    std::array<std::size_t, 3> triggered_pixels{{0, 0, 0}};
    // Measured directly via CUDA events (stage 0 start to stage 10 end), not a sum.
    double full_trigger_ms = 0.0;
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
};

class BatchTrigger
{
public:
    BatchTrigger(int width, int height, int gaussian_kernel_size,
                 float gaussian_sigma, float spark_cut,
                 float threshold_cut, int dilation_radius);
    ~BatchTrigger();

    BatchTrigger(const BatchTrigger&) = delete;
    BatchTrigger& operator=(const BatchTrigger&) = delete;

    void set_pedestals(const std::array<FloatImage, 3>& pedestals);
    void enqueue(const std::array<Image, 3>& images);
    void synchronize(std::array<Image, 3>& results, BatchTiming& timing);
    void synchronize_output(BatchTiming& timing);
    void write_output_pgm(const std::string& filename, int camera) const;
    void copy_centroid_mask(int camera, std::vector<uint8_t>& output);
    void copy_filtered_image(int camera, std::vector<float>& output);
    void copy_sparkless_image(int camera, std::vector<float>& output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Image read_pgm(const std::string& filename);
void write_pgm(const std::string& filename, const Image& image);
void write_pgm(const std::string& filename, const uint16_t* pixels,
               int width, int height);
