Cygno Trigger CUDA
==================

This directory contains the CUDA implementation of the image trigger. The
reference CPU implementation is located in ../pains_trigger and remains
separate.

The CUDA pipeline uses:

- CUDA kernels for thresholds, masks, and the final mask application;
- OpenCV CUDA for pedestal subtraction, Laplacian filtering, Gaussian filtering,
  and dilations;
- MIDAS to receive CAM banks from the SYSTEM buffer;
- ROOT only to load the pedestal map.

All installable components remain inside this directory. The project does not
install the NVIDIA driver and none of the commands below use sudo.

1. System requirements
----------------------

The machine must provide:

- an NVIDIA GPU and a working NVIDIA driver;
- CMake 3.22 or newer;
- a C++ compiler;
- Git;
- ROOT with ROOT::Hist;
- MIDAS with MIDASSYS configured;

The GPU used for this project is:

  NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  Compute capability: 12.0

CUDA Toolkit 12.8 or newer is recommended. CUDA 11.2, if present on the
machine, is too old for the definitive Blackwell build and should not be used.

Check the environment:

  nvidia-smi
  echo "$MIDASSYS"
  echo "$ROOTSYS"
  cmake --version
  g++ --version

2. Enter the project directory
------------------------------

  cd /home/standard/daq/pains_trigger_cuda

3. Install the CUDA Toolkit locally
------------------------------------

Download the official NVIDIA CUDA 12.8.1 installer into third_party/. The URL
below is the installer used for this project:

  mkdir -p third_party
  wget -c -O third_party/cuda_12.8.1_570.124.06_linux.run \
    https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run

Install only the toolkit into a local directory. Do not use the --driver option:

  chmod +x third_party/cuda_12.8.1_570.124.06_linux.run
  third_party/cuda_12.8.1_570.124.06_linux.run \
    --silent \
    --toolkit \
    --toolkitpath="$PWD/third_party/cuda-local"

Configure the local environment:

  export CUDA_HOME="$PWD/third_party/cuda-local"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

Confirm the installation:

  nvcc --version

The NVIDIA driver remains the system-wide driver. The toolkit, headers, nvcc,
and libcudart are installed in third_party/cuda-local/.

4. Download OpenCV and opencv_contrib
--------------------------------------

  git clone --branch 4.x --depth 1 \
    https://github.com/opencv/opencv.git third_party/opencv-src

  git clone --branch 4.x --depth 1 \
    https://github.com/opencv/opencv_contrib.git third_party/opencv-contrib-src

The opencv and opencv_contrib sources must use the same branch or tag. The
cudev module from opencv_contrib is required by the OpenCV CUDA modules.

5. Build OpenCV with local CUDA
--------------------------------

Configure OpenCV for the RTX PRO 6000 architecture, using a minimal module set
that contains the operations required by the trigger:

  cmake -S third_party/opencv-src \
    -B third_party/opencv-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/third_party/opencv-cuda-local" \
    -DOPENCV_EXTRA_MODULES_PATH="$PWD/third_party/opencv-contrib-src/modules" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDA_ARCH_BIN=12.0 \
    -DCUDA_ARCH_PTX=12.0 \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DWITH_NVCUVID=OFF \
    -DWITH_NVCUVENC=OFF \
    -DBUILD_LIST=core,imgproc,cudev,cudaarithm,cudafilters,cudaimgproc \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_TBB=OFF \
    -DWITH_IPP=ON \
    -DOPENCV_GENERATE_PKGCONFIG=ON

Use one build job if the machine has limited memory or parallel builds are
interrupted:

  cmake --build third_party/opencv-build -j1
  cmake --install third_party/opencv-build

With more memory, two jobs can be used:

  cmake --build third_party/opencv-build -j2

The CMake summary should contain:

  CUDA detected: 12.8
  CUDA: Using CUDA_ARCH_BIN=12.0
  NVIDIA CUDA: YES

The installation is created in:

  third_party/opencv-cuda-local/

6. Build the CUDA trigger
-------------------------

  export CUDA_HOME="$PWD/third_party/cuda-local"
  export PATH="$CUDA_HOME/bin:$PWD/tools:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PWD/third_party/opencv-cuda-local/lib:${LD_LIBRARY_PATH:-}"

  cmake -S . -B build \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DOpenCV_DIR="$PWD/third_party/opencv-cuda-local/lib/cmake/opencv4"

  cmake --build build -j1

The main executables are:

  build/CygnoTriggerCudaDatared
  build/CygnoTriggerCudaPgm
  build/CygnoTriggerCudaDebug

7. Configure the trigger
------------------------

Edit:

  nano config/configFile.txt

Recommended CAM1 configuration:

  pedestal_file=/home/standard/daq/pains_trigger/pedmaps/pedmap_run123601_rebin1.root
  pedestal_histogram=pedmap_1
  camera_id=1
  width=4096
  height=2304
  gaussian_kernel_size=27
  gaussian_sigma=8.0
  spark_cut=400.0
  threshold_cut=0.5
  dilation_radius=20
  inspect_only=false
  save_pairs_directory=comparison_images
  max_saved_pairs=5

For CAM0, change camera_id and pedestal_histogram to 0 and pedmap_0.

8. Run using MIDAS images
-------------------------

The live mode does not read PGM files from comparison_images. It receives CAM
banks directly from the SYSTEM buffer.

With MIDAS and the frontend running:

  ./run_trigger_cuda.sh

The launcher configures CUDA, OpenCV CUDA, the local ROOT compiler helper, and
runs build/CygnoTriggerCudaDatared.

The program exits after max_saved_pairs when save_pairs_directory is set. If
saving is disabled, it continues until Ctrl+C.

Output files use this format:

  comparison_images/original_CAM1_0.pgm
  comparison_images/triggered_CAM1_0.pgm

When image saving is enabled, the client also creates:

  comparison_images/log.txt
  comparison_images/log.csv

The human-readable log.txt contains the configuration used for the run and a
separate multi-line block for each processed event. Each block reports upload
time, pedestal subtraction,
Laplacian, spark threshold, spark dilation, spark mask, Gaussian filter,
centroid threshold, centroid dilation, final mask, download time, the total
GPU algorithm time, the complete GPU pipeline time, and the total processing
time including host conversion and PGM output. It also reports the number of
pixels retained in the triggered image.

The log.csv file contains only the numeric event table with no configuration
section, including the triggered_pixels column. It can be opened with
spreadsheet software or loaded with pandas.

9. inspect_only mode
--------------------

To inspect images received from MIDAS without loading the pedestal or applying
the trigger, set this in config/configFile.txt:

  inspect_only=true

In this mode the program connects to MIDAS, reads the configured CAM bank,
prints the number of pixels and their minimum, maximum, and nonzero counts, and
ignores the pedestal and saving options.

10. Offline test with the CPU reference frames
-----------------------------------------------

This test does not use MIDAS. It uses the five original PGM files saved by the
CPU version, so CPU and GPU process exactly the same inputs:

  ./build/CygnoTriggerCudaPgm \
    /home/standard/daq/pains_trigger/comparison_images \
    comparison_images \
    /home/standard/daq/pains_trigger/pedmaps/pedmap_run123601_rebin1.root \
    5

To investigate the first stage that differs:

  ./build/CygnoTriggerCudaDebug \
    /home/standard/daq/pains_trigger/comparison_images/original_CAM1_0.pgm \
    /home/standard/daq/pains_trigger/pedmaps/pedmap_run123601_rebin1.root

The debug executable compares pedestal subtraction, the Laplacian, spark masks,
the sparkless image, the Gaussian result, the centroid mask, and the final
dilation.

11. Measured performance
------------------------

For a 4096 x 2304 image on the RTX PRO 6000, after CUDA initialization:

  GPU algorithm:                         approximately 6 ms
  CPU -> GPU upload:                     approximately 11 ms
  GPU -> CPU download:                   approximately 68 ms
  complete pipeline with transfers:      approximately 85 ms

The first event can be slower because the CUDA context is initialized. PGM
reading and writing are not part of the online trigger algorithm timing.

The five reference frames were compared pixel by pixel:

  different_pixels=0
  max_difference=0
  sum_difference=0

12. Cleaning and rebuilding
---------------------------

Remove only the CUDA trigger build:

  rm -rf build

Remove the intermediate OpenCV build while preserving the installed OpenCV:

  rm -rf third_party/opencv-build

Rebuild OpenCV from source while preserving the downloaded sources:

  rm -rf third_party/opencv-build third_party/opencv-cuda-local

Then repeat sections 5 and 6.

13. Important files
-------------------

  CMakeLists.txt
  README.txt
  config/configFile.txt
  run_trigger_cuda.sh
  include/trigger_cuda.hpp
  include/root_io.hpp
  src/trigger_cuda.cu
  src/root_io.cpp
  src/datared_cuda.cpp
  src/pgm_client.cpp
  src/debug_compare.cpp
  third_party/cuda-local/
  third_party/opencv-src/
  third_party/opencv-contrib-src/
  third_party/opencv-cuda-local/

No step in this README installs the NVIDIA driver or system-wide libraries.
