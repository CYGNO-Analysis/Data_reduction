#include "root_io.hpp"
#include "trigger.hpp"

#include "midas.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <cctype>
#include <fstream>
#include <iostream>
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
    std::string pedestal_histogram = "pedmap";
    int camera_id = 1;
    int width = 0;
    int height = 0;
    bool inspect_only = false;
    std::string save_first_file;
    std::string save_pairs_directory;
    int max_saved_pairs = 5;
    int gaussian_kernel_size = 27;
    float gaussian_sigma = 8.0f;
    float spark_cut = 400.0f;
    float threshold_cut = 0.5f;
    int dilation_radius = 20;
};

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --width N --height N [--inspect-only] [options]\n"
              << "Options: --pedestal FILE --pedestal-hist NAME"
              << " --config FILE --camera-id N"
              << " --spark-cut N --threshold N --dilation-radius N"
              << " --save-first FILE"
              << " --save-pairs DIR (recommended: comparison_images/)\n";
}

Options parse_options(int argc, char** argv)
{
    Options options;
    std::string config_file = "config/configFile.txt";
    bool config_explicit = false;

    for (int index = 1; index + 1 < argc; ++index)
    {
        if (std::string(argv[index]) == "--config")
        {
            config_file = argv[index + 1];
            config_explicit = true;
            break;
        }
    }

    std::ifstream configuration(config_file);
    if (!configuration)
    {
        if (config_explicit)
            throw std::runtime_error("Could not open config file: " + config_file);
    }
    else
    {
        std::string line;
        while (std::getline(configuration, line))
        {
            const std::size_t comment = line.find('#');
            if (comment != std::string::npos)
                line.resize(comment);

            const std::size_t separator = line.find('=');
            if (separator == std::string::npos)
                continue;

            auto trim = [](std::string value) {
                value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                    [](unsigned char character) { return !std::isspace(character); }));
                value.erase(std::find_if(value.rbegin(), value.rend(),
                    [](unsigned char character) { return !std::isspace(character); }).base(),
                    value.end());
                return value;
            };

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
            else if (key == "save_first_file") options.save_first_file = value;
            else if (key == "save_pairs_directory") options.save_pairs_directory = value;
            else if (key == "max_saved_pairs") options.max_saved_pairs = std::stoi(value);
            else if (key == "inspect_only") options.inspect_only = value == "true" || value == "1";
        }
    }

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];

        auto value = [&](const char* name) -> std::string
        {
            if (index + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++index];
        };

        if (argument == "--config")
            ++index;
        else if (argument == "--pedestal")
            options.pedestal_file = value("--pedestal");
        else if (argument == "--pedestal-hist")
            options.pedestal_histogram = value("--pedestal-hist");
        else if (argument == "--camera-id")
            options.camera_id = std::stoi(value("--camera-id"));
        else if (argument == "--width")
            options.width = std::stoi(value("--width"));
        else if (argument == "--height")
            options.height = std::stoi(value("--height"));
        else if (argument == "--inspect-only")
            options.inspect_only = true;
        else if (argument == "--save-first")
            options.save_first_file = value("--save-first");
        else if (argument == "--save-pairs")
            options.save_pairs_directory = value("--save-pairs");
        else if (argument == "--spark-cut")
            options.spark_cut = std::stof(value("--spark-cut"));
        else if (argument == "--threshold")
            options.threshold_cut = std::stof(value("--threshold"));
        else if (argument == "--dilation-radius")
            options.dilation_radius = std::stoi(value("--dilation-radius"));
        else
            throw std::runtime_error("Unknown option: " + argument);
    }

    if (options.camera_id < 0 || options.camera_id > 9)
        throw std::runtime_error("camera_id must be between 0 and 9");
    if (options.max_saved_pairs <= 0)
        throw std::runtime_error("max_saved_pairs must be positive");
    if (!options.inspect_only && (options.width <= 0 || options.height <= 0))
        throw std::runtime_error("--width and --height are required unless --inspect-only is used");
    if (!options.inspect_only && options.pedestal_file.empty())
        throw std::runtime_error("--pedestal is required unless --inspect-only is used");
    if (!options.save_first_file.empty() && (options.width <= 0 || options.height <= 0))
        throw std::runtime_error("--width and --height are required with --save-first");
    if (!options.save_pairs_directory.empty()
        && (options.width <= 0 || options.height <= 0))
        throw std::runtime_error("--width and --height are required with --save-pairs");
    if (options.inspect_only)
    {
        options.save_first_file.clear();
        options.save_pairs_directory.clear();
    }

    return options;
}

void write_pgm(const std::string& filename, const Image& image)
{
    std::ofstream output(filename, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create image file: " + filename);

    output << "P5\n" << image.width << ' ' << image.height << "\n65535\n";
    for (int value : image.data)
    {
        const uint16_t pixel = static_cast<uint16_t>(
            std::max(0, std::min(65535, value))
        );
        output.put(static_cast<char>(pixel >> 8));
        output.put(static_cast<char>(pixel & 0xff));
    }
}

void write_pgm(const std::string& filename, const uint16_t* source,
               int width, int height)
{
    Image image(width, height);
    for (std::size_t index = 0; index < image.data.size(); ++index)
        image.data[index] = source[index];
    write_pgm(filename, image);
}

void ensure_directory(const std::string& directory)
{
    if (mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("Could not create output directory: " + directory);
}

void initialize_logs(const Options& options)
{
    if (options.save_pairs_directory.empty())
        return;

    std::ofstream text_log(options.save_pairs_directory + "/log.txt");
    std::ofstream csv_log(options.save_pairs_directory + "/log.csv");
    if (!text_log || !csv_log)
        throw std::runtime_error("Could not create trigger logs");

    csv_log << "event,pedestal_ms,laplacian_ms,spark_threshold_ms,"
        << "spark_dilation_ms,spark_mask_ms,gaussian_ms,"
        << "centroid_threshold_ms,centroid_dilation_ms,final_mask_ms,"
        << "triggered_pixels,algorithm_total_ms,cpu_pipeline_total_ms,"
        << "total_processing_ms\n";
}

void process_camera(const BANK32& bank, const void* data, const Options& options,
                    const FloatImage* pedestal, unsigned int event,
                    int& saved_pairs)
{
    if (bank.data_size % sizeof(uint16_t) != 0)
        throw std::runtime_error("CAM bank has a non-16-bit payload");

    const std::size_t pixels = bank.data_size / sizeof(uint16_t);
    const std::size_t expected = static_cast<std::size_t>(options.width) * options.height;
    if (!options.inspect_only && pixels != expected)
        throw std::runtime_error("CAM bank size does not match --width x --height");

    const auto* source = static_cast<const uint16_t*>(data);

    const char selected_camera = static_cast<char>('0' + options.camera_id);
    if (bank.name[3] != selected_camera)
        return;

    if (!options.save_first_file.empty() && event == 1)
    {
        std::ofstream image_file(options.save_first_file, std::ios::binary);
        if (!image_file)
            throw std::runtime_error("Could not create image file: " + options.save_first_file);

        image_file << "P5\n" << options.width << ' ' << options.height << "\n65535\n";
        for (std::size_t index = 0; index < pixels; ++index)
        {
            const unsigned char high = static_cast<unsigned char>(source[index] >> 8);
            const unsigned char low = static_cast<unsigned char>(source[index] & 0xff);
            image_file.put(static_cast<char>(high));
            image_file.put(static_cast<char>(low));
        }
    }

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

        std::cout << "Event " << event << " " << std::string(bank.name, 4)
                  << ": pixels=" << pixels
                  << ", min=" << minimum
                  << ", max=" << maximum
                  << ", nonzero=" << nonzero << '\n';
        return;
    }

    const auto processing_start = std::chrono::steady_clock::now();
    Image image(options.width, options.height);
    for (std::size_t index = 0; index < pixels; ++index)
        image.data[index] = source[index];

    TriggerTiming timing;
    const auto trigger_start = std::chrono::steady_clock::now();
    Image triggered = trigger(image, *pedestal, options.gaussian_kernel_size,
                              options.gaussian_sigma, options.spark_cut,
                              options.threshold_cut, options.dilation_radius,
                              &timing);
    const auto trigger_end = std::chrono::steady_clock::now();
    const double trigger_time_ms = std::chrono::duration<double, std::milli>(
        trigger_end - trigger_start
    ).count();

    if (!options.save_pairs_directory.empty())
    {
        const std::string index = std::to_string(saved_pairs);
        write_pgm(options.save_pairs_directory + "/original_CAM"
                  + std::to_string(options.camera_id) + "_" + index + ".pgm",
                  source, options.width, options.height);
        write_pgm(options.save_pairs_directory + "/triggered_CAM"
                  + std::to_string(options.camera_id) + "_" + index + ".pgm",
                  triggered);
        ++saved_pairs;
    }

    const double total_processing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start
    ).count();
    const std::size_t triggered_pixels = timing.triggered_pixels;
    if (!options.save_pairs_directory.empty())
    {
        std::ofstream text_log(options.save_pairs_directory + "/log.txt", std::ios::app);
        std::ofstream csv_log(options.save_pairs_directory + "/log.csv", std::ios::app);
        if (!text_log || !csv_log)
            throw std::runtime_error("Could not append to trigger logs");

        text_log << "Event " << event << " CAM" << options.camera_id << '\n'
            << "triggered_pixels=" << triggered_pixels << '\n'
            << "  Pedestal subtraction: " << timing.pedestal_ms << " ms\n"
            << "  Laplacian: " << timing.spark_laplacian_ms << " ms\n"
            << "  Spark threshold: " << timing.spark_threshold_ms << " ms\n"
            << "  Spark dilation: " << timing.spark_dilation_ms << " ms\n"
            << "  Spark mask: " << timing.spark_mask_ms << " ms\n"
            << "  Gaussian: " << timing.gaussian_ms << " ms\n"
            << "  Centroid threshold: " << timing.centroid_threshold_ms << " ms\n"
            << "  Centroid dilation: " << timing.centroid_dilation_ms << " ms\n"
            << "  Final mask: " << timing.final_mask_ms << " ms\n"
            << "  Algorithm total: " << trigger_time_ms << " ms\n"
            << "  CPU pipeline total: " << trigger_time_ms << " ms\n"
            << "  Total processing: " << total_processing_ms << " ms\n\n";

        csv_log << event << ','
            << timing.pedestal_ms << ','
            << timing.spark_laplacian_ms << ','
            << timing.spark_threshold_ms << ','
            << timing.spark_dilation_ms << ','
            << timing.spark_mask_ms << ','
            << timing.gaussian_ms << ','
            << timing.centroid_threshold_ms << ','
            << timing.centroid_dilation_ms << ','
            << timing.final_mask_ms << ','
            << triggered_pixels << ','
            << trigger_time_ms << ','
            << trigger_time_ms << ','
            << total_processing_ms << '\n';
    }

    std::cout << "Event " << event << " " << std::string(bank.name, 4)
              << ": algorithm=" << trigger_time_ms << " ms"
              << ", triggered_pixels=" << triggered_pixels << '\n';
}
}

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = parse_options(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        FloatImage pedestal;
        if (!options.inspect_only)
        {
            pedestal = load_pedestal_root(options.pedestal_file,
                                          options.pedestal_histogram);
            if (pedestal.width != options.width || pedestal.height != options.height)
                throw std::runtime_error("Pedestal dimensions do not match camera dimensions");
        }
        if (!options.save_pairs_directory.empty())
        {
            ensure_directory(options.save_pairs_directory);
            initialize_logs(options);
        }

        struct sigaction signal_action{};
        signal_action.sa_handler = signal_handler;
        sigemptyset(&signal_action.sa_mask);
        sigaction(SIGINT, &signal_action, nullptr);
        sigaction(SIGTERM, &signal_action, nullptr);

        HNDLE buffer = 0;
        if (cm_connect_experiment(nullptr, nullptr, "cygno_trigger", nullptr) != CM_SUCCESS)
            throw std::runtime_error("Connection to MIDAS experiment failed");
        if (cm_set_watchdog_params(TRUE, 120000) != CM_SUCCESS)
            throw std::runtime_error("Could not configure MIDAS watchdog");
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

        while (running
               && (options.save_pairs_directory.empty()
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
            BANK_HEADER* bank_header = reinterpret_cast<BANK_HEADER*>(
                event.data() + sizeof(EVENT_HEADER));
            BANK32* bank = nullptr;
            void* data = nullptr;

            if (bk_iterate32(bank_header, &bank, &data))
            {
                do
                {
                    if (bank->name[0] == 'C' && bank->name[1] == 'A'
                        && bank->name[2] == 'M')
                        process_camera(*bank, data, options,
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