#pragma once

#include <string>

#include "trigger_cuda.hpp"

FloatImage load_pedestal_root(const std::string& filename, const std::string& hist_name);
