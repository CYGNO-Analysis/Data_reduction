Cygno Trigger CPU
==================

This directory contains the CPU implementation of the Cygno image trigger.
The CUDA implementation is maintained separately in ../pains_trigger_cuda.

The trigger uses OpenCV CPU functions for pedestal subtraction, Laplacian
filtering, thresholds, dilations, Gaussian filtering, and mask application.
CUDA is not required or enabled in this project.

The online client receives CAM banks directly from the MIDAS SYSTEM buffer. ROOT
is used only to load the pedestal map. The original CPU algorithm and the
OpenCV implementation were validated against the same five CAM1 frames.

1. External prerequisites
-------------------------

The machine must provide:

- a C++ compiler and CMake 3.16 or newer;
- ROOT with ROOT::Hist;
- MIDAS with MIDASSYS configured;
- Git;

Check the environment:

  nvidia-smi
  echo "$MIDASSYS"
  echo "$ROOTSYS"
  cmake --version
  g++ --version

CUDA and an NVIDIA GPU are not required for this CPU version.

2. Enter the project directory
------------------------------

  cd /home/standard/daq/pains_trigger

3. Prepare the ROOT compiler workaround
----------------------------------------

The ROOT installation on this machine searches for a compiler named
x86_64-linux-gnu-g++-9. If that executable does not exist, create the local
alias below. This changes nothing outside pains_trigger.

  mkdir -p tools
  ln -sfn "$(command -v x86_64-linux-gnu-g++-11)" \
    tools/x86_64-linux-gnu-g++-9

If the machine uses another available compiler, replace the target path with
that compiler.

4. Download OpenCV source locally
----------------------------------

Run this only if third_party/opencv-src does not exist:

  mkdir -p third_party
  git clone --branch 4.x --depth 1 \
    https://github.com/opencv/opencv.git third_party/opencv-src

The CPU build needs only the OpenCV core and imgproc modules. No opencv_contrib,
CUDA Toolkit, or global OpenCV installation is required.

5. Build and install OpenCV locally
------------------------------------

The OpenCV installation is local to pains_trigger and does not require sudo.
Run these commands if third_party/opencv-local does not exist, or if OpenCV
must be rebuilt:

  cmake -S third_party/opencv-src \
    -B third_party/opencv-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/third_party/opencv-local" \
    -DBUILD_LIST=core,imgproc \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_TBB=OFF \
    -DWITH_IPP=ON

  cmake --build third_party/opencv-build -j2
  cmake --install third_party/opencv-build

Use -j1 instead of -j2 if the machine has limited memory or parallel builds are
interrupted:

  cmake --build third_party/opencv-build -j1

The result is installed in:

  third_party/opencv-local/

6. Build the CPU trigger
------------------------

The project requires the local OpenCV installation:

  cmake -S . -B build \
    -DOpenCV_DIR="$PWD/third_party/opencv-local/lib/cmake/opencv4"

  cmake --build build --target CygnoTriggerDatared -j2

The executable is generated at:

  build/CygnoTriggerDatared

The CMake project builds only the MIDAS client. The CPU pipeline is linked from:

  src/trigger.cpp
  src/root_io.cpp

7. Configure the trigger
------------------------

Edit the configuration file:

  nano config/configFile.txt

Recommended CAM1 configuration:

  pedestal_file=pedmaps/pedmap_run123601_rebin1.root
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

The pedestal file used by this project contains pedmap_0 and pedmap_1, each with
4096 x 2304 bins. The C++ loader stores values using data[y * width + x]. This
matches the camera bank layout and corresponds to uproot values().T in Python.
Do not apply another transpose in C++.

8. Run using MIDAS images
-------------------------

MIDAS and the frontend must be running and publishing CAM banks in the SYSTEM
buffer. The recommended command is:

  ./run_trigger.sh

The launcher configures the local ROOT compiler helper and OpenCV runtime path,
then runs build/CygnoTriggerDatared. It uses paths relative to its own directory.

The equivalent direct command, executed from pains_trigger, is:

  PATH="$PWD/tools:$PATH" \
  LD_LIBRARY_PATH="$PWD/third_party/opencv-local/lib:${LD_LIBRARY_PATH:-}" \
  ./build/CygnoTriggerDatared

The program reads config/configFile.txt automatically. Stop it with Ctrl+C,
unless max_saved_pairs is reached.

9. inspect_only mode
--------------------

To inspect images received from MIDAS without loading the pedestal or applying
the trigger, set this in config/configFile.txt:

  inspect_only=true

In this mode the program:

- connects to MIDAS;
- reads the configured CAM bank;
- prints pixel count, minimum, maximum, and nonzero count;
- does not load the pedestal;
- does not call the trigger;
- ignores save_pairs_directory, max_saved_pairs, and save_first_file.

This is useful for checking the image dimensions and raw values before enabling
the trigger.

10. Save a limited number of comparisons
-----------------------------------------

Set these values in config/configFile.txt:

  inspect_only=false
  save_pairs_directory=comparison_images
  max_saved_pairs=5

Then run:

  ./run_trigger.sh

The program saves at most five original and five triggered 16-bit PGM images and
then exits. For CAM1, the files are named:

  comparison_images/original_CAM1_0.pgm
  comparison_images/triggered_CAM1_0.pgm

The same directory also receives:

  comparison_images/log.txt
  comparison_images/log.csv

The human-readable log.txt contains the trigger configuration and a separate
multi-line block for each event. It reports every trigger stage, the algorithm
total, the CPU pipeline total, and the total processing time including host
conversion and PGM output. It also reports the number of pixels retained in the
triggered image.

The log.csv file contains only the numeric event table, including the
triggered_pixels column. It has no configuration section and can be opened with
spreadsheet software or loaded with pandas.

PGM is used because it preserves the original 16-bit pixel values without lossy
compression. The files can be copied to another machine and loaded with NumPy.

11. CPU algorithm and measured performance
-------------------------------------------

The trigger applies these stages in order:

1. subtract the pedestal;
2. calculate the custom 3x3 Laplacian;
3. threshold the Laplacian to identify sparks;
4. dilate the spark mask with radius 1;
5. zero spark pixels in the pedestal-subtracted image;
6. apply a Gaussian filter with kernel 27 and sigma 8;
7. threshold the filtered image at 0.5;
8. dilate the centroid mask with radius 20;
9. copy selected pixels from the original image to the output.

For a 4096 x 2304 image, the OpenCV CPU implementation was measured at roughly:

  algorithm total:         approximately 418-530 ms
  CPU pipeline total:      approximately 418-530 ms
  Gaussian filter:         approximately 65-75 ms
  Laplacian:               approximately 65-75 ms
  spark dilation:          approximately 54-80 ms
  centroid dilation:       approximately 55-80 ms

The CPU pipeline total is the trigger time and does not include PGM output. The
total processing time in log.txt additionally includes conversion of the MIDAS
buffer and writing the original and triggered PGM files. The exact time depends
on machine load and whether image PGM I/O is included. These values do not
include MIDAS event acquisition.

12. CPU/GPU validation and cleanup
----------------------------------

The CPU outputs can be compared against the CUDA outputs using the same original
PGM frames. The CPU reference files are stored in:

  comparison_images/

The CUDA implementation and its validation tools are in:

  ../pains_trigger_cuda/

The five tested CPU/GPU outputs were pixel-identical:

  different_pixels=0
  max_difference=0
  sum_difference=0

To remove only generated CPU trigger build files:

  rm -rf build

To remove the intermediate OpenCV build while preserving the installed local
OpenCV:

  rm -rf third_party/opencv-build

To rebuild OpenCV from source, remove both the build directory and the local
installation, then repeat sections 4 and 5:

  rm -rf third_party/opencv-build third_party/opencv-local

13. Important files
-------------------

Source code:

  include/trigger.hpp
  include/root_io.hpp
  src/trigger.cpp
  src/root_io.cpp
  src/datared_client.cpp

Configuration and launcher:

  config/configFile.txt
  run_trigger.sh

Pedestal:

  pedmaps/pedmap_run123601_rebin1.root

OpenCV source and local CPU installation:

  third_party/opencv-src/
  third_party/opencv-local/

Runtime helper:

  tools/x86_64-linux-gnu-g++-9

No CUDA Toolkit or global OpenCV installation is needed for this CPU version.
