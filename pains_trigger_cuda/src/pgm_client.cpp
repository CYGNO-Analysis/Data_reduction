#include "root_io.hpp"
#include "trigger_cuda.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " INPUT_DIRECTORY OUTPUT_DIRECTORY PEDESTAL_ROOT [COUNT]\n";
        return 1;
    }

    try
    {
        const std::string input_directory = argv[1];
        const std::string output_directory = argv[2];
        const std::string pedestal_file = argv[3];
        const int count = argc > 4 ? std::stoi(argv[4]) : 5;
        const FloatImage pedestal = load_pedestal_root(pedestal_file, "pedmap_1");

        for (int index = 0; index < count; ++index)
        {
            const auto event_start = std::chrono::steady_clock::now();
            const std::string suffix = std::to_string(index);
            const Image image = read_pgm(
                input_directory + "/original_CAM1_" + suffix + ".pgm"
            );
            TriggerTiming timing;
            const Image triggered = trigger_cuda(
                image, pedestal, 27, 8.0f, 400.0f, 0.5f, 20, &timing
            );
            write_pgm(
                output_directory + "/triggered_CAM1_" + suffix + ".pgm",
                triggered
            );

            const double algorithm_ms = timing.pedestal_ms
                + timing.laplacian_ms + timing.spark_threshold_ms
                + timing.spark_dilation_ms + timing.spark_mask_ms
                + timing.gaussian_ms + timing.centroid_threshold_ms
                + timing.centroid_dilation_ms + timing.final_mask_ms;
            const double gpu_pipeline_ms = timing.upload_ms + algorithm_ms
                + timing.download_ms;
            const double event_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - event_start
            ).count();

            std::cout << "Event " << index << '\n'
                      << "  upload=" << timing.upload_ms << " ms\n"
                      << "  pedestal=" << timing.pedestal_ms << " ms\n"
                      << "  laplacian=" << timing.laplacian_ms << " ms\n"
                      << "  spark_threshold=" << timing.spark_threshold_ms << " ms\n"
                      << "  spark_dilation=" << timing.spark_dilation_ms << " ms\n"
                      << "  spark_mask=" << timing.spark_mask_ms << " ms\n"
                      << "  gaussian=" << timing.gaussian_ms << " ms\n"
                      << "  centroid_threshold=" << timing.centroid_threshold_ms << " ms\n"
                      << "  centroid_dilation=" << timing.centroid_dilation_ms << " ms\n"
                      << "  final_mask=" << timing.final_mask_ms << " ms\n"
                      << "  download=" << timing.download_ms << " ms\n"
                      << "  algorithm_total=" << algorithm_ms << " ms\n"
                      << "  gpu_pipeline_total=" << gpu_pipeline_ms << " ms\n"
                      << "  event_total_including_pgm_io=" << event_ms << " ms\n";
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
