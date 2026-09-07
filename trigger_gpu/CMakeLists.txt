cmake_minimum_required(VERSION 3.22)

project(CygnoTriggerCuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Hist)
find_package(Midas REQUIRED PATHS $ENV{MIDASSYS})
find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaarithm cudafilters cudaimgproc)

add_library(trigger_cuda src/trigger_cuda.cu src/trigger_context.cu src/root_io.cpp)
target_include_directories(trigger_cuda PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trigger_cuda PUBLIC ${OpenCV_LIBS} CUDA::cudart ROOT::Hist)

add_executable(CygnoTriggerCudaDatared src/datared_cuda.cpp)
target_link_libraries(CygnoTriggerCudaDatared PRIVATE trigger_cuda midas::mfed midas::drivers)
