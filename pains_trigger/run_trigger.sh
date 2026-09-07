#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PATH="$ROOT_DIR/tools:$PATH"
export LD_LIBRARY_PATH="$ROOT_DIR/third_party/opencv-local/lib:${LD_LIBRARY_PATH:-}"

exec "$ROOT_DIR/build/CygnoTriggerDatared"
