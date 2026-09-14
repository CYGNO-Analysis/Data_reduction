cmake_minimum_required(VERSION 3.22)

project(CygnoTriggerCudaBatch LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Hist)
find_package(Midas REQUIRED PATHS $ENV{MIDASSYS})
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(trigger_cuda_batch src/batch_trigger.cu src/root_io.cpp)
target_include_directories(trigger_cuda_batch PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trigger_cuda_batch PUBLIC ${OpenCV_LIBS} CUDA::cudart ROOT::Hist)

add_executable(CygnoTriggerCudaBatch src/datared_batch.cpp)
target_link_libraries(CygnoTriggerCudaBatch PRIVATE trigger_cuda_batch midas::mfed midas::drivers)