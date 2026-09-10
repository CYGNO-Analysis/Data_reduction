Cygno Trigger GPU Batch
========================

Experimental batch implementation for CAM0, CAM1, and CAM2. The three cameras
are stored as planes in one GPU buffer with layout [camera][height][width]. A
single CUDA pipeline processes all three planes. Pedestals are uploaded once
and remain resident on the GPU.

Requirements
------------

- NVIDIA driver and GPU;
- CUDA Toolkit 12.8 or newer;
- ROOT with ROOT::Hist;
- MIDAS with MIDASSYS configured;
- CMake 3.22 or newer;
- OpenCV CUDA 4.x built with the local CUDA toolkit;
- the ROOT and MIDAS development installations available through `ROOTSYS` and
  `MIDASSYS`.

Repository layout
-----------------

The current working copy shares CUDA and OpenCV with `trigger_gpu` through the
`third_party` symlink. The shared installation is located at the repository
root, not inside either trigger directory. Therefore, downloading only this
directory does not
provide those dependencies. The recommended setup is to clone the complete
repository:

  git clone https://github.com/ifpains/Data_reduction.git
  cd Data_reduction/trigger_gpu_batch

The commands below then use:

  ../third_party/cuda-local/
  ../third_party/opencv-cuda-local/

For a standalone copy, replace the `third_party` symlink with a real directory
and install CUDA and OpenCV into it as described below. The CUDA installer,
OpenCV sources, opencv_contrib sources, and the resulting OpenCV build are not
stored in this batch directory.

The pedestal ROOT file must contain pedmap_0, pedmap_1, and pedmap_2, each with
4096 x 2304 bins. The current configuration uses:

  ../pedmaps/pedmap_run123685_rebin1.root

Build
-----

If using the complete repository, the existing local CUDA/OpenCV installations
are reused through the `third_party` symlink. Verify that the symlink resolves:

  readlink -f third_party
  test -x third_party/cuda-local/bin/nvcc
  test -f third_party/opencv-cuda-local/lib/cmake/opencv4/OpenCVConfig.cmake

For a standalone copy, install CUDA Toolkit 12.8 locally and build OpenCV CUDA
4.x with `core,imgproc` and the CUDA modules `cudev,cudaarithm,cudafilters,
cudaimgproc`. The installation must end up at:

  third_party/cuda-local/
  third_party/opencv-cuda-local/

The CUDA installer must be run with `--toolkit` and
`--toolkitpath="$PWD/third_party/cuda-local"`, without `--driver`. OpenCV must
be configured with `CUDA_TOOLKIT_ROOT_DIR="$PWD/third_party/cuda-local"`,
`CMAKE_CUDA_COMPILER="$PWD/third_party/cuda-local/bin/nvcc"`, and CUDA
architecture 12.0 for the RTX PRO 6000 used here.

With either setup, configure from this directory:

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

The executable is build/CygnoTriggerCudaBatch.

Run
---

With MIDAS publishing CAM0, CAM1, and CAM2 in the SYSTEM buffer:

  ./run_trigger.sh

The program rejects events missing any of the three camera banks or having the
wrong payload dimensions. Output images are written to ../comparison_images as
original_CAM0_N.pgm, triggered_CAM0_N.pgm, and the equivalent CAM1/CAM2 files.
Logs use log_gpu_batch.txt and log_gpu_batch.csv.

The online print uses the same timing vocabulary as trigger_gpu:

  algorithm
  upload
  download
  full_trigger_total
  full_trigger_sum_check
  total

Here `algorithm` and `full_trigger_total` refer to the three-camera batch. The
`total` includes the host bank copies and, when enabled, PGM output for all
three cameras.

`full_trigger_total` is measured directly with CUDA events, from the start of
the upload stage to the end of the download stage, so it correctly captures
any GPU work that happens between stages, including mask application and
pixel counting. `full_trigger_sum_check` is the arithmetic sum of upload +
algorithm total + mask apply + download, kept only as a cross-check against
the directly measured value.

The human-readable log_gpu_batch.txt additionally reports a `Mask apply` line,
separate from `Download`. It covers the GPU kernel that applies the final
centroid mask to the original three-camera image (producing the final
uint16_t image) and the kernel that counts triggered pixels per camera. Before
this stage was measured separately, it was launched between the last
algorithm stage and the download stage without a dedicated CUDA event pair,
so it was silently missing from `full_trigger_sum_check`; the gap was
approximately 2.2-2.8 ms per event on this GPU. Splitting it into its own
stage closed that gap to CUDA-event measurement noise (well under 0.01 ms).

The input images are uploaded as uint16_t. The GPU applies the final mask and
downloads the three triggered images as uint16_t in a persistent pinned output
buffer. The online client writes those downloaded `uint16_t` images directly
to PGM and does not convert them to the internal `Image` format. The persistent
batch input and output buffers use CUDA pinned host memory (`cudaMallocHost`).

Algorithm layout
----------------

Each stage launches one CUDA kernel over 3 * width * height pixels. The
Laplacian, separable Gaussian, thresholds, spark correction, separable
morphological dilations, final mask, and per-camera pixel counts all operate on
the batch layout without crossing camera boundaries.
