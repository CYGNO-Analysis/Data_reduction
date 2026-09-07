cmake_minimum_required(VERSION 3.16)

project(CygnoTrigger LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ROOT REQUIRED COMPONENTS Hist)
find_package(Midas REQUIRED PATHS $ENV{MIDASSYS})
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(trigger src/trigger.cpp src/root_io.cpp)

target_include_directories(trigger PUBLIC include)

target_link_libraries(trigger PUBLIC ROOT::Hist)
target_include_directories(trigger PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trigger PUBLIC ${OpenCV_LIBS})

add_executable(CygnoTriggerDatared src/datared_client.cpp)

target_link_libraries(CygnoTriggerDatared PRIVATE trigger midas::mfed midas::drivers)