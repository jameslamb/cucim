#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"
package_src_dir="${package_dir}/src/${package_name}"

CMAKE_BUILD_TYPE="release"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

rapids-logger "Generating build requirements"
matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee -a /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# First build the C++ lib using CMake via the run script
./run build_local all ${CMAKE_BUILD_TYPE}

sccache --show-adv-stats

cd "${package_dir}"

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*
ls -1 final_dist | grep -vqz 'none'

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
