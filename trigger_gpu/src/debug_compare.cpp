#include "root_io.hpp"
#include "trigger_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
struct CpuDebug
{
    TriggerDebug values;
};

FloatImage copy_float(const cv::Mat& matrix)
{
    FloatImage result(matrix.cols, matrix.rows);
    std::copy(reinterpret_cast<const float*>(matrix.data),
              reinterpret_cast<const float*>(matrix.data) + result.data.size(),
              result.data.begin());
    return result;
}

std::vector<uint8_t> copy_mask(const cv::Mat& matrix)
{
    return std::vector<uint8_t>(matrix.data, matrix.data + matrix.total());
}

CpuDebug run_cpu(const Image& image, const FloatImage& pedestal)
{
    CpuDebug debug;
    cv::Mat image_mat(image.height, image.width, CV_32S,
                      const_cast<int*>(image.data.data()));
    cv::Mat pedestal_mat(pedestal.height, pedestal.width, CV_32F,
                         const_cast<float*>(pedestal.data.data()));
    cv::Mat pedsub;
    cv::subtract(image_mat, pedestal_mat, pedsub, cv::noArray(), CV_32F);
    debug.values.pedestal_subtracted = copy_float(pedsub);

    const cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f);
    cv::Mat laplacian;
    cv::filter2D(pedsub, laplacian, CV_32F, kernel, cv::Point(-1, -1),
                 0.0, cv::BORDER_CONSTANT);
    debug.values.laplacian = copy_float(laplacian);

    cv::Mat spark_mask;
    cv::compare(laplacian, 400.0f, spark_mask, cv::CMP_GE);
    debug.values.spark_mask = copy_mask(spark_mask);

    cv::Mat spark_dilated;
    cv::dilate(spark_mask, spark_dilated, cv::Mat::ones(3, 3, CV_8U),
               cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
               cv::morphologyDefaultBorderValue());
    debug.values.spark_dilated = copy_mask(spark_dilated);

    cv::Mat sparkless = pedsub.clone();
    sparkless.setTo(0.0f, spark_dilated);
    debug.values.sparkless = copy_float(sparkless);

    cv::Mat filtered;
    cv::GaussianBlur(sparkless, filtered, cv::Size(27, 27), 8.0, 8.0,
                     cv::BORDER_CONSTANT);
    debug.values.filtered = copy_float(filtered);

    cv::Mat centroid_mask;
    cv::compare(filtered, 0.5f, centroid_mask, cv::CMP_GE);
    debug.values.centroid_mask = copy_mask(centroid_mask);

    cv::Mat centroid_dilated;
    cv::dilate(centroid_mask, centroid_dilated, cv::Mat::ones(41, 41, CV_8U),
               cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
               cv::morphologyDefaultBorderValue());
    debug.values.centroid_dilated = copy_mask(centroid_dilated);
    return debug;
}

void compare_float(const char* name, const FloatImage& cpu, const FloatImage& gpu)
{
    double max_difference = 0.0;
    std::size_t different = 0;
    double sum = 0.0;
    std::size_t printed = 0;
    for (std::size_t index = 0; index < cpu.data.size(); ++index)
    {
        const double difference = std::abs(static_cast<double>(cpu.data[index])
                                           - gpu.data[index]);
        max_difference = std::max(max_difference, difference);
        sum += difference;
        different += difference > 0.0;
        if (difference > 0.0 && printed < 8)
        {
            const int x = static_cast<int>(index % cpu.width);
            const int y = static_cast<int>(index / cpu.width);
            std::cout << "  " << name << " mismatch (x=" << x << ", y=" << y
                      << "): cpu=" << cpu.data[index]
                      << ", gpu=" << gpu.data[index] << '\n';
            ++printed;
        }
    }
    std::cout << name << ": different=" << different
              << ", max_abs=" << max_difference << ", sum_abs=" << sum << '\n';
}

void compare_mask(const char* name, const std::vector<uint8_t>& cpu,
                  const std::vector<uint8_t>& gpu)
{
    std::size_t different = 0;
    std::size_t cpu_only = 0;
    std::size_t gpu_only = 0;
    for (std::size_t index = 0; index < cpu.size(); ++index)
    {
        const bool cpu_value = cpu[index] != 0;
        const bool gpu_value = gpu[index] != 0;
        different += cpu_value != gpu_value;
        cpu_only += cpu_value && !gpu_value;
        gpu_only += gpu_value && !cpu_value;
    }
    std::cout << name << ": different=" << different
              << ", cpu_only=" << cpu_only << ", gpu_only=" << gpu_only << '\n';
}
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " IMAGE.pgm PEDESTAL.root\n";
        return 1;
    }

    const Image image = read_pgm(argv[1]);
    const FloatImage pedestal = load_pedestal_root(argv[2], "pedmap_1");
    TriggerDebug gpu;
    trigger_cuda_debug(image, pedestal, 27, 8.0f, 400.0f, 0.5f, 20, gpu);
    const CpuDebug cpu = run_cpu(image, pedestal);

    compare_float("pedestal_subtracted", cpu.values.pedestal_subtracted, gpu.pedestal_subtracted);
    compare_float("laplacian", cpu.values.laplacian, gpu.laplacian);
    compare_mask("spark_mask", cpu.values.spark_mask, gpu.spark_mask);
    compare_mask("spark_dilated", cpu.values.spark_dilated, gpu.spark_dilated);
    compare_float("sparkless", cpu.values.sparkless, gpu.sparkless);
    compare_float("filtered", cpu.values.filtered, gpu.filtered);
    compare_mask("centroid_mask", cpu.values.centroid_mask, gpu.centroid_mask);
    compare_mask("centroid_dilated", cpu.values.centroid_dilated, gpu.centroid_dilated);
    return 0;
}
