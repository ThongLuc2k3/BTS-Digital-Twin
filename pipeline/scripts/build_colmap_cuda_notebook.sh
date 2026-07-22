#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PIPELINE_DIR")"

COLMAP_REPO_URL="${COLMAP_REPO_URL:-https://github.com/colmap/colmap.git}"
COLMAP_REF="${COLMAP_REF:-main}"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-/tmp/colmap-src}"
COLMAP_BUILD_DIR="${COLMAP_BUILD_DIR:-/tmp/colmap-build}"
COLMAP_INSTALL_PREFIX="${COLMAP_INSTALL_PREFIX:-$PROJECT_ROOT/.local/colmap-cuda}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-auto}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<EOF
Build COLMAP with CUDA support for headless notebook environments.

Environment variables:
  COLMAP_REF                   git ref to build (default: main)
  COLMAP_SRC_DIR               clone dir (default: /tmp/colmap-src)
  COLMAP_BUILD_DIR             build dir (default: /tmp/colmap-build)
  COLMAP_INSTALL_PREFIX        install dir (default: $PROJECT_ROOT/.local/colmap-cuda)
  CMAKE_CUDA_ARCHITECTURES     CUDA arch, e.g. 75 for Tesla T4 (default: native)
  INSTALL_DEPS                 1 to apt-install dependencies (default: 1)
  INSTALL_CUDA_TOOLKIT         auto|1|0 (default: auto)
  BUILD_JOBS                   parallel build jobs (default: nproc)

Example:
  CMAKE_CUDA_ARCHITECTURES=75 bash pipeline/scripts/build_colmap_cuda_notebook.sh
EOF
  exit 0
fi

run_root() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

install_deps() {
  local install_cuda=0

  if [[ "$INSTALL_CUDA_TOOLKIT" == "1" ]]; then
    install_cuda=1
  elif [[ "$INSTALL_CUDA_TOOLKIT" == "auto" ]] && ! command -v nvcc >/dev/null 2>&1; then
    install_cuda=1
  fi

  run_root apt-get update
  run_root apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libqt6svg6-dev \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev
  run_root mkdir -p /usr/include/opencv4

  if [[ "$install_cuda" == "1" ]]; then
    run_root apt-get install -y \
      nvidia-cuda-toolkit \
      nvidia-cuda-toolkit-gcc
  fi
}

maybe_set_gcc10() {
  if [[ ! -f /etc/os-release ]]; then
    return
  fi

  # COLMAP's install docs call out Ubuntu 22.04 + default CUDA package as a GCC mismatch case.
  # If gcc-10 exists, prefer it automatically to reduce notebook build friction.
  local version_id=""
  version_id="$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")"
  if [[ "$version_id" == "22.04" ]] && command -v gcc-10 >/dev/null 2>&1 && command -v g++-10 >/dev/null 2>&1; then
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export CUDAHOSTCXX=/usr/bin/g++-10
  fi
}

clone_or_update() {
  if [[ -d "$COLMAP_SRC_DIR/.git" ]]; then
    git -C "$COLMAP_SRC_DIR" fetch --depth 1 origin "$COLMAP_REF"
    git -C "$COLMAP_SRC_DIR" checkout --force FETCH_HEAD
  else
    rm -rf "$COLMAP_SRC_DIR"
    git clone --depth 1 --branch "$COLMAP_REF" "$COLMAP_REPO_URL" "$COLMAP_SRC_DIR"
  fi
}

build_colmap() {
  rm -rf "$COLMAP_BUILD_DIR"
  mkdir -p "$COLMAP_BUILD_DIR" "$COLMAP_INSTALL_PREFIX"

  cmake -S "$COLMAP_SRC_DIR" -B "$COLMAP_BUILD_DIR" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$COLMAP_INSTALL_PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DBLA_VENDOR=Intel10_64lp

  cmake --build "$COLMAP_BUILD_DIR" --parallel "$BUILD_JOBS"
  cmake --install "$COLMAP_BUILD_DIR"
}

verify_cuda() {
  local colmap_bin="$COLMAP_INSTALL_PREFIX/bin/colmap"
  if [[ ! -x "$colmap_bin" ]]; then
    echo "Lỗi: không thấy binary $colmap_bin" >&2
    exit 1
  fi

  local help_out
  help_out="$("$colmap_bin" -h 2>&1 || true)"
  printf '%s\n' "$help_out"
  if grep -q "without CUDA" <<<"$help_out"; then
    echo "Lỗi: build xong nhưng COLMAP vẫn without CUDA" >&2
    exit 1
  fi
}

if [[ "$INSTALL_DEPS" == "1" ]]; then
  install_deps
fi

maybe_set_gcc10
clone_or_update
build_colmap
verify_cuda

echo
echo "COLMAP CUDA build hoàn tất."
echo "COLMAP_BIN=$COLMAP_INSTALL_PREFIX/bin/colmap"
echo "Ví dụ chạy B2:"
echo "  export COLMAP_BIN=$COLMAP_INSTALL_PREFIX/bin/colmap"
echo "  bash pipeline/scripts/05_run_b2_pilot.sh hcm0031"
