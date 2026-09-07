#include "root_io.hpp"
#include "trigger_batch.hpp"

#include "midas.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

namespace
{
volatile std::sig_atomic_t running = 1;

void signal_handler(int)
{
    running = 0;
}

struct Options
{
    std::string pedestal_file;
    std::array<std::string, 3> pedestal_histogram{{"pedmap_0", "pedmap_1", "pedmap_2"}};
    int width = 4096;
    int height = 2304;
    int gaussian_kernel_size = 27;
    float gaussian_sigma = 8.0f;
    float spark_cut = 400.0f;
    float threshold_cut = 0.5f;
    int dilation_radius = 20;
    bool inspect_only = false;
    std::string save_pairs_directory;
    int max_saved_sets = 5;
};

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    const auto last = value.find_last_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return value.substr(first, last - first + 1);
}

Options load_config(const std::string& filename)
{
    Options options;
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Could not open config file: " + filename);
    std::string line;
    while (std::getline(input, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.resize(comment);
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            continue;
        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key == "pedestal_file") options.pedestal_file = value;
        else if (key == "pedestal_histogram_cam0") options.pedestal_histogram[0] = value;
        else if (key == "pedestal_histogram_cam1") options.pedestal_histogram[1] = value;
        else if (key == "pedestal_histogram_cam2") options.pedestal_histogram[2] = value;
        else if (key == "width") options.width = std::stoi(value);
        else if (key == "height") options.height = std::stoi(value);
        else if (key == "gaussian_kernel_size") options.gaussian_kernel_size = std::stoi(value);
        else if (key == "gaussian_sigma") options.gaussian_sigma = std::stof(value);
        else if (key == "spark_cut") options.spark_cut = std::stof(value);
        else if (key == "threshold_cut") options.threshold_cut = std::stof(value);
        else if (key == "dilation_radius") options.dilation_radius = std::stoi(value);
        else if (key == "inspect_only") options.inspect_only = value == "true" || value == "1";
        else if (key == "save_pairs_directory") options.save_pairs_directory = value;
        else if (key == "max_saved_sets") options.max_saved_sets = std::stoi(value);
    }
    if (options.width <= 0 || options.height <= 0)
        throw std::runtime_error("Image dimensions must be positive");
    if (!options.inspect_only && options.pedestal_file.empty())
        throw std::runtime_error("pedestal_file is required");
    if (options.max_saved_sets <= 0)
        throw std::runtime_error("max_saved_sets must be positive");
    if (options.inspect_only)
        options.save_pairs_directory.clear();
    return options;
}

void ensure_directory(const std::string& directory)
{
    if (mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("Could not create output directory: " + directory);
}

void initialize_log(const Options& options)
{
    if (options.save_pairs_directory.empty())
        return;
    std::ofstream text(options.save_pairs_directory + "/log_gpu_batch.txt");
    std::ofstream csv(options.save_pairs_directory + "/log_gpu_batch.csv");
    if (!text || !csv)
        throw std::runtime_error("Could not create batch logs");
    csv << "event,upload_ms,pedestal_ms,laplacian_ms,spark_threshold_ms,"
        << "spark_dilation_ms,spark_mask_ms,gaussian_ms,centroid_threshold_ms,"
        << "centroid_dilation_ms,download_ms,algorithm_total_ms,"
        << "full_trigger_total_ms,total_ms,"
        << "camera0_pixels,camera1_pixels,camera2_pixels\n";
}

bool is_camera_bank(const BANK32& bank, int camera)
{
    const char expected[4] = {'C', 'A', 'M', static_cast<char>('0' + camera)};
    return std::equal(std::begin(expected), std::end(expected), bank.name);
}

void append_log(const Options& options, unsigned int event, const BatchTiming& timing,
                double total_ms)
{
    if (options.save_pairs_directory.empty())
        return;
    std::ofstream text(options.save_pairs_directory + "/log_gpu_batch.txt", std::ios::app);
    std::ofstream csv(options.save_pairs_directory + "/log_gpu_batch.csv", std::ios::app);
    if (!text || !csv)
        throw std::runtime_error("Could not append batch logs");
    const double algorithm_ms = timing.pedestal_ms + timing.laplacian_ms
        + timing.spark_threshold_ms + timing.spark_dilation_ms
        + timing.spark_mask_ms + timing.gaussian_ms
        + timing.centroid_threshold_ms + timing.centroid_dilation_ms;
    const double full_trigger_ms = timing.upload_ms + algorithm_ms
        + timing.download_ms;
        text << "Event " << event << "\n"
            << "  Upload: " << timing.upload_ms << " ms\n"
            << "  Pedestal subtraction: " << timing.pedestal_ms << " ms\n"
            << "  Laplacian: " << timing.laplacian_ms << " ms\n"
            << "  Spark threshold: " << timing.spark_threshold_ms << " ms\n"
            << "  Spark dilation: " << timing.spark_dilation_ms << " ms\n"
            << "  Spark mask: " << timing.spark_mask_ms << " ms\n"
            << "  Gaussian: " << timing.gaussian_ms << " ms\n"
            << "  Centroid threshold: " << timing.centroid_threshold_ms << " ms\n"
            << "  Centroid dilation: " << timing.centroid_dilation_ms << " ms\n"
            << "  Download: " << timing.download_ms << " ms\n"
             << "  Algorithm total: " << algorithm_ms << " ms\n"
             << "  Full trigger total: " << full_trigger_ms << " ms\n"
             << "  GPU batch: " << timing.gpu_ms << " ms\n"
             << "  Total processing: " << total_ms << " ms\n"
         << "  triggered_pixels: " << timing.triggered_pixels[0] << ", "
         << timing.triggered_pixels[1] << ", " << timing.triggered_pixels[2] << "\n\n";
    csv << event << ',' << timing.upload_ms << ',' << timing.pedestal_ms << ','
        << timing.laplacian_ms << ',' << timing.spark_threshold_ms << ','
        << timing.spark_dilation_ms << ',' << timing.spark_mask_ms << ','
        << timing.gaussian_ms << ',' << timing.centroid_threshold_ms << ','
        << timing.centroid_dilation_ms << ',' << timing.download_ms << ','
        << algorithm_ms << ',' << full_trigger_ms << ',' << total_ms << ','
        << timing.triggered_pixels[0] << ',' << timing.triggered_pixels[1] << ','
        << timing.triggered_pixels[2] << '\n';
}
}

int main(int argc, char** argv)
{
    const std::string config_file = argc > 1 ? argv[1] : "config/configFile.txt";
    try
    {
        const Options options = load_config(config_file);
        std::array<FloatImage, 3> pedestals;
        if (!options.inspect_only)
        {
            for (int camera = 0; camera < 3; ++camera)
            {
                std::cout << "Loading CAM" << camera << " pedestal '"
                          << options.pedestal_histogram[camera] << "'\n";
                pedestals[camera] = load_pedestal_root(
                    options.pedestal_file, options.pedestal_histogram[camera]);
                if (pedestals[camera].width != options.width
                    || pedestals[camera].height != options.height)
                    throw std::runtime_error("Pedestal dimensions do not match CAM"
                        + std::to_string(camera));
            }
        }

        BatchTrigger trigger(options.width, options.height, options.gaussian_kernel_size,
                             options.gaussian_sigma, options.spark_cut,
                             options.threshold_cut, options.dilation_radius);
        if (!options.inspect_only)
            trigger.set_pedestals(pedestals);
        if (!options.save_pairs_directory.empty())
        {
            ensure_directory(options.save_pairs_directory);
            initialize_log(options);
        }

        struct sigaction signal_action{};
        signal_action.sa_handler = signal_handler;
        sigemptyset(&signal_action.sa_mask);
        sigaction(SIGINT, &signal_action, nullptr);
        sigaction(SIGTERM, &signal_action, nullptr);

        if (cm_connect_experiment(nullptr, nullptr, "cygno_trigger_cuda_batch", nullptr) != CM_SUCCESS)
            throw std::runtime_error("Connection to MIDAS experiment failed");
        HNDLE buffer = 0;
        if (bm_open_buffer("SYSTEM", 1000000000, &buffer) != BM_SUCCESS)
            throw std::runtime_error("Could not open MIDAS SYSTEM buffer");
        INT request = 0;
        if (bm_request_event(buffer, 1, TRIGGER_ALL, GET_RECENT | GET_NONBLOCKING,
                             &request, nullptr) != BM_SUCCESS)
            throw std::runtime_error("Could not register MIDAS event request");

        constexpr std::size_t event_capacity = 1000 * 1024 * 1024;
        std::vector<char> event(event_capacity);
        unsigned int event_number = 0;
        int saved_sets = 0;
        while (running && (options.save_pairs_directory.empty()
                           || saved_sets < options.max_saved_sets))
        {
            INT event_size = static_cast<INT>(event.size());
            const INT status = bm_receive_event(buffer, event.data(), &event_size, BM_NO_WAIT);
            cm_yield(1);
            if (status == BM_ASYNC_RETURN)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (status != BM_SUCCESS)
                throw std::runtime_error("Error receiving MIDAS event");

            ++event_number;
            std::array<const BANK32*, 3> banks{{nullptr, nullptr, nullptr}};
            std::array<const void*, 3> data{{nullptr, nullptr, nullptr}};
            auto* bank_header = reinterpret_cast<BANK_HEADER*>(
                event.data() + sizeof(EVENT_HEADER));
            BANK32* bank = nullptr;
            void* bank_data = nullptr;
            if (bk_iterate32(bank_header, &bank, &bank_data))
            {
                do
                {
                    for (int camera = 0; camera < 3; ++camera)
                        if (is_camera_bank(bank[0], camera))
                        {
                            banks[camera] = &bank[0];
                            data[camera] = bank_data;
                        }
                }
                while (bk_iterate32(bank_header, &bank, &bank_data));
            }
            for (int camera = 0; camera < 3; ++camera)
                if (!banks[camera])
                    throw std::runtime_error("Event " + std::to_string(event_number)
                        + " is missing CAM" + std::to_string(camera));

            const std::size_t expected = static_cast<std::size_t>(options.width)
                * options.height * sizeof(uint16_t);
            const auto start = std::chrono::steady_clock::now();
            std::array<Image, 3> images{
                Image(options.width, options.height), Image(options.width, options.height),
                Image(options.width, options.height)};
            for (int camera = 0; camera < 3; ++camera)
            {
                if (banks[camera]->data_size != expected)
                    throw std::runtime_error("CAM" + std::to_string(camera)
                        + " payload size does not match dimensions");
                const auto* source = static_cast<const uint16_t*>(data[camera]);
                for (std::size_t index = 0; index < images[camera].data.size(); ++index)
                    images[camera].data[index] = source[index];
            }
            if (options.inspect_only)
                continue;

            trigger.enqueue(images);
            BatchTiming timing;
            trigger.synchronize_output(timing);
            const double gpu_algorithm_ms = timing.pedestal_ms + timing.laplacian_ms
                + timing.spark_threshold_ms + timing.spark_dilation_ms
                + timing.spark_mask_ms + timing.gaussian_ms
                + timing.centroid_threshold_ms + timing.centroid_dilation_ms;
            const double full_trigger_ms = timing.upload_ms + gpu_algorithm_ms
                + timing.download_ms;
            if (!options.save_pairs_directory.empty())
            {
                const std::string index = std::to_string(saved_sets);
                for (int camera = 0; camera < 3; ++camera)
                {
                    write_pgm(options.save_pairs_directory + "/original_CAM"
                        + std::to_string(camera) + "_" + index + ".pgm", images[camera]);
                    trigger.write_output_pgm(options.save_pairs_directory + "/triggered_CAM"
                        + std::to_string(camera) + "_" + index + ".pgm", camera);
                }
            }
            const double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            append_log(options, event_number, timing, total_ms);
            std::cout << "Event " << event_number << ": algorithm=" << gpu_algorithm_ms
                      << " ms, upload=" << timing.upload_ms
                      << " ms, download=" << timing.download_ms
                      << " ms, full_trigger_total=" << full_trigger_ms
                      << " ms, total=" << total_ms << " ms"
                      << ", triggered_pixels=[CAM0=" << timing.triggered_pixels[0]
                      << ", CAM1=" << timing.triggered_pixels[1]
                      << ", CAM2=" << timing.triggered_pixels[2] << "]\n";
            ++saved_sets;
        }
        bm_delete_request(request);
        bm_close_buffer(buffer);
        cm_disconnect_experiment();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        cm_disconnect_experiment();
        return 1;
    }
    return 0;
}
