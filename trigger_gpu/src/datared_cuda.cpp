#include "root_io.hpp"
#include "trigger_cuda.hpp"

#include "midas.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
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
    std::string pedestal_histogram = "pedmap_1";
    int camera_id = 1;
    int width = 4096;
    int height = 2304;
    int gaussian_kernel_size = 27;
    float gaussian_sigma = 8.0f;
    float spark_cut = 400.0f;
    float threshold_cut = 0.5f;
    int dilation_radius = 20;
    bool inspect_only = false;
    std::string save_pairs_directory;
    int max_saved_pairs = 5;
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
        else if (key == "pedestal_histogram") options.pedestal_histogram = value;
        else if (key == "camera_id") options.camera_id = std::stoi(value);
        else if (key == "width") options.width = std::stoi(value);
        else if (key == "height") options.height = std::stoi(value);
        else if (key == "gaussian_kernel_size") options.gaussian_kernel_size = std::stoi(value);
        else if (key == "gaussian_sigma") options.gaussian_sigma = std::stof(value);
        else if (key == "spark_cut") options.spark_cut = std::stof(value);
        else if (key == "threshold_cut") options.threshold_cut = std::stof(value);
        else if (key == "dilation_radius") options.dilation_radius = std::stoi(value);
        else if (key == "inspect_only") options.inspect_only = value == "true" || value == "1";
        else if (key == "save_pairs_directory") options.save_pairs_directory = value;
        else if (key == "max_saved_pairs") options.max_saved_pairs = std::stoi(value);
    }

    if (options.inspect_only)
        options.save_pairs_directory.clear();
    if (options.camera_id < 0 || options.camera_id > 9)
        throw std::runtime_error("camera_id must be between 0 and 9");
    if (options.width <= 0 || options.height <= 0)
        throw std::runtime_error("Image dimensions must be positive");
    if (!options.inspect_only && options.pedestal_file.empty())
        throw std::runtime_error("pedestal_file is required");
    if (options.max_saved_pairs <= 0)
        throw std::runtime_error("max_saved_pairs must be positive");
    return options;
}

void ensure_directory(const std::string& directory)
{
    if (mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("Could not create output directory: " + directory);
}

void save_pgm(const std::string& filename, const Image& image)
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

void initialize_log(const Options& options)
{
    if (options.save_pairs_directory.empty())
        return;

    std::ofstream text_log(options.save_pairs_directory + "/log_gpu.txt");
    std::ofstream csv_log(options.save_pairs_directory + "/log_gpu.csv");
    if (!text_log || !csv_log)
        throw std::runtime_error("Could not create trigger logs");

    csv_log << "event,upload_ms,pedestal_ms,laplacian_ms,spark_threshold_ms,"
            << "spark_dilation_ms,spark_mask_ms,gaussian_ms,"
            << "centroid_threshold_ms,centroid_dilation_ms,"
            << "mask_apply_ms,download_ms,triggered_pixels,algorithm_total_ms,"
            << "full_trigger_total_ms,full_trigger_sum_check_ms,total_processing_ms\n";
}

void process_camera(const BANK32& bank, const void* data, const Options& options,
                    const FloatImage* pedestal, unsigned int event,
                    int& saved_pairs)
{
    const char expected_name[4] = {'C', 'A', 'M', static_cast<char>('0' + options.camera_id)};
    if (!std::equal(std::begin(expected_name), std::end(expected_name), bank.name))
        return;
    if (bank.data_size % sizeof(uint16_t) != 0)
        throw std::runtime_error("CAM bank has a non-16-bit payload");

    const std::size_t pixels = bank.data_size / sizeof(uint16_t);
    const std::size_t expected_pixels = static_cast<std::size_t>(options.width) * options.height;
    if (pixels != expected_pixels)
        throw std::runtime_error("CAM bank size does not match configured dimensions");

    const auto* source = static_cast<const uint16_t*>(data);
    if (options.inspect_only)
    {
        uint16_t minimum = source[0];
        uint16_t maximum = source[0];
        std::size_t nonzero = 0;
        for (std::size_t index = 0; index < pixels; ++index)
        {
            minimum = std::min(minimum, source[index]);
            maximum = std::max(maximum, source[index]);
            nonzero += source[index] != 0;
        }
        std::cout << "Event " << event << " CAM" << options.camera_id
                  << ": pixels=" << pixels << ", min=" << minimum
                  << ", max=" << maximum << ", nonzero=" << nonzero << '\n';
        return;
    }

    const auto processing_start = std::chrono::steady_clock::now();
    Image image(options.width, options.height);
    for (std::size_t index = 0; index < pixels; ++index)
        image.data[index] = source[index];

    static std::unique_ptr<TriggerContext> trigger_context;
    if (!trigger_context)
        trigger_context.reset(new TriggerContext(
            *pedestal, options.gaussian_kernel_size, options.gaussian_sigma,
            options.spark_cut, options.threshold_cut, options.dilation_radius));

    TriggerTiming timing;
    Image triggered;
    std::string triggered_filename;
    if (!options.save_pairs_directory.empty() && saved_pairs < options.max_saved_pairs)
    {
        const std::string index = std::to_string(saved_pairs);
        triggered_filename = options.save_pairs_directory + "/triggered_CAM"
            + std::to_string(options.camera_id) + "_" + index + ".pgm";
        trigger_context->process_gpu_output_pgm(image, triggered_filename, &timing);
    }
    else
        triggered = trigger_context->process_gpu_output(image, &timing);

    const double algorithm_ms = timing.pedestal_ms + timing.laplacian_ms
        + timing.spark_threshold_ms + timing.spark_dilation_ms + timing.spark_mask_ms
        + timing.gaussian_ms + timing.centroid_threshold_ms
        + timing.centroid_dilation_ms;
    // Cross-check only: sum of individually timed stages, not a direct measurement.
    const double full_trigger_sum_ms = timing.upload_ms + algorithm_ms
        + timing.mask_apply_ms + timing.download_ms;
    const double full_trigger_ms = timing.full_trigger_ms;
    const std::size_t triggered_pixels = timing.triggered_pixels;
    if (!options.save_pairs_directory.empty() && saved_pairs < options.max_saved_pairs)
    {
        const std::string index = std::to_string(saved_pairs);
        save_pgm(options.save_pairs_directory + "/original_CAM"
                      + std::to_string(options.camera_id) + "_" + index + ".pgm", image);
        if (triggered_filename.empty())
            save_pgm(options.save_pairs_directory + "/triggered_CAM"
                          + std::to_string(options.camera_id) + "_" + index + ".pgm", triggered);
        ++saved_pairs;
        std::cout << "Saved pair " << saved_pairs << "/"
                  << options.max_saved_pairs << '\n';
    }

    const double total_processing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start
    ).count();
    std::cout << "Event " << event << " CAM" << options.camera_id
              << ": algorithm=" << algorithm_ms << " ms"
              << ", upload=" << timing.upload_ms << " ms"
              << ", download=" << timing.download_ms << " ms"
              << ", full_trigger_total=" << full_trigger_ms << " ms"
              << ", full_trigger_sum_check=" << full_trigger_sum_ms << " ms"
              << ", total=" << total_processing_ms << " ms"
              << ", triggered_pixels=" << triggered_pixels << '\n';
    if (!options.save_pairs_directory.empty())
    {
        std::ofstream text_log(options.save_pairs_directory + "/log_gpu.txt", std::ios::app);
        std::ofstream csv_log(options.save_pairs_directory + "/log_gpu.csv", std::ios::app);
        if (!text_log || !csv_log)
            throw std::runtime_error("Could not append to trigger logs");

        text_log << "Event " << event << " CAM" << options.camera_id << '\n'
            << "triggered_pixels=" << triggered_pixels << '\n'
            << "  Upload: " << timing.upload_ms << " ms\n"
            << "  Pedestal subtraction: " << timing.pedestal_ms << " ms\n"
            << "  Laplacian: " << timing.laplacian_ms << " ms\n"
            << "  Spark threshold: " << timing.spark_threshold_ms << " ms\n"
            << "  Spark dilation: " << timing.spark_dilation_ms << " ms\n"
            << "  Spark mask: " << timing.spark_mask_ms << " ms\n"
            << "  Gaussian: " << timing.gaussian_ms << " ms\n"
            << "  Centroid threshold: " << timing.centroid_threshold_ms << " ms\n"
            << "  Centroid dilation: " << timing.centroid_dilation_ms << " ms\n"
            << "  Mask apply: " << timing.mask_apply_ms << " ms\n"
            << "  Download: " << timing.download_ms << " ms\n"
            << "  Algorithm total: " << algorithm_ms << " ms\n"
            << "  Full trigger total: " << full_trigger_ms << " ms\n"
            << "  Full trigger (sum check): " << full_trigger_sum_ms << " ms\n"
            << "  Total processing: " << total_processing_ms << " ms\n\n";

        csv_log << event << ','
            << timing.upload_ms << ','
            << timing.pedestal_ms << ','
            << timing.laplacian_ms << ','
            << timing.spark_threshold_ms << ','
            << timing.spark_dilation_ms << ','
            << timing.spark_mask_ms << ','
            << timing.gaussian_ms << ','
            << timing.centroid_threshold_ms << ','
            << timing.centroid_dilation_ms << ','
            << timing.mask_apply_ms << ','
            << timing.download_ms << ','
            << triggered_pixels << ','
            << algorithm_ms << ','
            << full_trigger_ms << ','
            << full_trigger_sum_ms << ','
            << total_processing_ms << '\n';
    }
}

FloatImage load_configured_pedestal(const Options& options)
{
    if (options.inspect_only)
        return FloatImage();
    return load_pedestal_root(options.pedestal_file, options.pedestal_histogram);
}
}

int main(int argc, char** argv)
{
    const std::string config_file = argc > 1 ? argv[1] : "config/configFile.txt";
    try
    {
        const Options options = load_config(config_file);
        const FloatImage pedestal = load_configured_pedestal(options);
        if (!options.inspect_only)
        {
            if (pedestal.width != options.width || pedestal.height != options.height)
                throw std::runtime_error("Pedestal dimensions do not match camera dimensions");
        }
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

        if (cm_connect_experiment(nullptr, nullptr, "cygno_trigger_cuda", nullptr) != CM_SUCCESS)
            throw std::runtime_error("Connection to MIDAS experiment failed");
        if (cm_set_watchdog_params(TRUE, 120000) != CM_SUCCESS)
            throw std::runtime_error("Could not configure MIDAS watchdog");

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
        int saved_pairs = 0;
        while (running && (options.save_pairs_directory.empty()
                           || saved_pairs < options.max_saved_pairs))
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
            auto* bank_header = reinterpret_cast<BANK_HEADER*>(
                event.data() + sizeof(EVENT_HEADER));
            BANK32* bank = nullptr;
            void* data = nullptr;
            if (bk_iterate32(bank_header, &bank, &data))
            {
                do
                {
                    process_camera(bank[0], data, options,
                                   options.inspect_only ? nullptr : &pedestal,
                                   event_number, saved_pairs);
                }
                while (bk_iterate32(bank_header, &bank, &data));
            }
        }

        bm_delete_request(request);
        bm_close_buffer(buffer);
        cm_disconnect_experiment();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        cm_disconnect_experiment();
        return 1;
    }
}
