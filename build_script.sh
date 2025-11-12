#!/bin/bash

set -e
set -x

######## User Configurations ########
# Source directory
caesar_src_dir=.
# Build directory
caesar_build_dir=./build

export CMAKE_PREFIX_PATH=$HOME/Software/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/Software/nvcomp/lib:$LD_LIBRARY_PATH

mkdir -p ${caesar_build_dir}    
cmake -S ${caesar_src_dir} -B ${caesar_build_dir} \
    -DCMAKE_PREFIX_PATH="$HOME/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/share/cmake" \
    -DCMAKE_CXX_FLAGS="-I$HOME/Software/nvcomp/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/Software/nvcomp/lib" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTS=ON

cmake --build ${caesar_build_dir} --config Release -- -j$(nproc)