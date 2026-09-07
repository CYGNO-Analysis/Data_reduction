Cygno Trigger CUDA
==================

This directory contains the CUDA implementation of the image trigger. The
reference CPU implementation is located in ../trigger_cpu and remains
separate.

The CUDA pipeline uses:

- CUDA kernels for thresholds, masks, and the final mask application;
- OpenCV CUDA for pedestal subtraction, Laplacian filtering, Gaussian filtering,
  and dilations;
- MIDAS to receive CAM banks from the SYSTEM buffer;
- ROOT only to load the pedestal map.

The shared CUDA and OpenCV installations are kept in the repository-level
`../third_party/` directory and are used by both GPU implementations. The
project does not install the NVIDIA driver and none of the commands below use
sudo.

1. System requirements
----------------------

The machine must provide:

- an NVIDIA GPU and a working NVIDIA driver;
- CMake 3.22 or newer;
- a C++ compiler;
- Git;
- ROOT with ROOT::Hist;
- MIDAS with MIDASSYS configured;

ROOT and MIDAS are external dependencies; they are not installed by this
README or included in the repository. Before configuring CMake, load the
environment provided by those installations and verify their locations:

  export ROOTSYS=/path/to/root
  export MIDASSYS=/path/to/midas
  test -x "$ROOTSYS/bin/root-config"
  test -f "$MIDASSYS/include/midas.h"

The GPU used for this project is:

  NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  Compute capability: 12.0

CUDA Toolkit 12.8 or newer is recommended. CUDA 11.2, if present on the
machine, is too old for the definitive Blackwell build and should not be used.

The pedestal ROOT file is an external input and is not tracked in Git. Place
it at:

  ../pedmaps/pedmap_run123601_rebin1.root

It must contain the TH2 histograms pedmap_0 and pedmap_1 with 4096 x 2304
bins.

Check the environment:

  nvidia-smi
  echo "$MIDASSYS"
  echo "$ROOTSYS"
  cmake --version
  g++ --version

2. Enter the project directory
------------------------------

  cd /path/to/Data_reduction/trigger_gpu

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

If the ROOT installation expects the local compiler helper, create it before
configuring CMake:

  mkdir -p tools
  ln -sfn "$(command -v x86_64-linux-gnu-g++-11)" \
    tools/x86_64-linux-gnu-g++-9

  export CUDA_HOME="$PWD/third_party/cuda-local"
  export PATH="$CUDA_HOME/bin:$PWD/tools:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PWD/third_party/opencv-cuda-local/lib:${LD_LIBRARY_PATH:-}"

  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DOpenCV_DIR="$PWD/third_party/opencv-cuda-local/lib/cmake/opencv4"

  cmake --build build -j1

The executable is:

  build/CygnoTriggerCudaDatared

The client uses a bandwidth-reduced transfer path:

- the camera image is uploaded to the GPU as `uint16_t` (`CV_16U`, 16-bit
  unsigned pixels);
- the GPU produces the final binary mask as `uint8_t` (`CV_8U`), with values 0
  or 255. It is not a C++ `bool` image;
- the GPU applies the mask to the original `uint16_t` image;
- the final triggered image is downloaded to the CPU as `uint16_t`;
- the CPU only converts the downloaded image to the internal `Image` format;
- `triggered_pixels` is counted from the resulting image.

The pedestal remains a `float` image on the GPU. The online client uses this
GPU-output path, so production timing has no CPU image reconstruction or output
conversion step.

The online client creates a persistent trigger context at startup. The three
pedestal, image, mask, and intermediate GPU buffers and all CUDA filters are
allocated once; the pedestal is uploaded once and reused for every event. Each
event uploads only the new camera image, runs the trigger, creates the final
triggered image on the GPU, and downloads that image.

The persistent host-side uint16 input buffer uses CUDA page-locked (pinned)
memory allocated with `cudaMallocHost`. This avoids allocating ordinary
pageable memory for every event and allows the upload path to use CUDA's
optimized host transfer mechanism.

The host-side output buffer for the final triggered `uint16_t` image is also
allocated once with `cudaMallocHost` and reused for every event. This avoids
reallocating the download destination and removes the initial download
allocator warmup from subsequent events.

The context also owns one persistent CUDA stream. Upload, trigger stages, and
mask download are enqueued in order on that stream and synchronized only once
before CPU output conversion. CUDA events measure the individual stages without
introducing a host synchronization after every stage.

The Gaussian stage uses a separable CUDA implementation with horizontal and
vertical passes, matching the configured Gaussian kernel and constant-zero
border behavior. This removes the per-event OpenCV CUDA Gaussian filter call.

The spark and centroid dilations also use separable CUDA horizontal and
vertical passes with constant-zero borders. Their configured radii are 1 and
20 respectively, so no OpenCV CUDA morphology filter is created per event.

7. Configure the trigger
------------------------

Edit:

  nano config/configFile.txt

Recommended CAM1 configuration:

  pedestal_file=../pedmaps/pedmap_run123685_rebin1.root
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
  save_pairs_directory=../comparison_images
  max_saved_pairs=5

For CAM0, change camera_id and pedestal_histogram to 0 and pedmap_0.

8. Run using MIDAS images
-------------------------

The live mode does not read PGM files from comparison_images. It receives CAM
banks directly from the SYSTEM buffer.

With MIDAS and the frontend running:

  ./run_trigger.sh

The launcher configures CUDA, OpenCV CUDA, the local ROOT compiler helper, and
runs build/CygnoTriggerCudaDatared.

The program exits after max_saved_pairs when save_pairs_directory is set. If
saving is disabled, it continues until Ctrl+C.

Output files use this format:

  ../comparison_images/original_CAM1_0.pgm
  ../comparison_images/triggered_CAM1_0.pgm

When image saving is enabled, the client also creates:

  ../comparison_images/log_gpu.txt
  ../comparison_images/log_gpu.csv

The human-readable log_gpu.txt contains the configuration used for the run and a
separate multi-line block for each processed event. Each block reports upload
time, pedestal subtraction, Laplacian, spark threshold, spark dilation, spark
mask, Gaussian filter, centroid threshold, centroid dilation, mask download,
CPU output conversion, GPU algorithm total, GPU pipeline total, and total
processing time including host conversion and PGM output. `full_trigger_total`
is upload plus the GPU algorithm plus image download, so it represents the
complete trigger before PGM/log I/O. It also reports the number of pixels
retained in the triggered image.

The log_gpu.csv file contains only the numeric event table with no configuration
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

10. Measured performance
------------------------

For a 4096 x 2304 image on the RTX PRO 6000, after CUDA initialization, the
optimized path is typically in these ranges:

  GPU algorithm:                         approximately 6-10 ms
  CPU -> GPU upload (uint16 image):      approximately 10-30 ms
  GPU -> CPU mask download (uint8):      approximately 1-8 ms
  CPU image reconstruction:              approximately 30-40 ms

The first event can be slower because the CUDA context, GPU buffers, and
filters are initialized. The exact values depend on memory state and system
load. The `total_processing_ms` value additionally includes host conversion
and PGM output when image saving is enabled.

11. Cleaning and rebuilding
---------------------------

Remove only the CUDA trigger build:

  rm -rf build

Remove the intermediate OpenCV build while preserving the installed OpenCV:

  rm -rf third_party/opencv-build

Rebuild OpenCV from source while preserving the downloaded sources:

  rm -rf third_party/opencv-build third_party/opencv-cuda-local

Then repeat sections 5 and 6.

12. Important files
-------------------

  CMakeLists.txt
  README.txt
  config/configFile.txt
  run_trigger.sh
  include/trigger_cuda.hpp
  include/root_io.hpp
  src/trigger_cuda.cu
  src/root_io.cpp
  src/datared_cuda.cpp
  third_party/cuda-local/
  third_party/opencv-src/
  third_party/opencv-contrib-src/
  third_party/opencv-cuda-local/

No step in this README installs the NVIDIA driver or system-wide libraries.
