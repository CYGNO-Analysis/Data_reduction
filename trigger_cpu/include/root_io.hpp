#pragma once

#include <string>

#include "trigger.hpp"

Image load_image_root(const std::string& filename, const std::string& hist_name);
FloatImage load_pedestal_root(const std::string& filename, const std::string& hist_name);