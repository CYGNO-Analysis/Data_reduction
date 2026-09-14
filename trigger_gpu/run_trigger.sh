#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export CUDA_HOME="$ROOT_DIR/third_party/cuda-local"
export PATH="$CUDA_HOME/bin:$ROOT_DIR/tools:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$ROOT_DIR/third_party/opencv-cuda-local/lib:${LD_LIBRARY_PATH:-}"

exec "$ROOT_DIR/build/CygnoTriggerCudaDatared" "$ROOT_DIR/config/configFile.txt"
