find_package(MPI)
find_package(CUDA)
find_package(PkgConfig)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# ZSTD Detection Logic
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
endif()

if(NOT ZSTD_FOUND)
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY NAMES aocl_compression zstd)
    
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        set(ZSTD_FOUND TRUE)
        message(STATUS "Found Zstd: ${ZSTD_LIBRARY}")
        
        add_library(zstd_target UNKNOWN IMPORTED)
        set_target_properties(zstd_target PROPERTIES
            IMPORTED_LOCATION "${ZSTD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Zstd library not found. Please install libzstd-dev or set CMAKE_PREFIX_PATH")
    endif()
else()
    message(STATUS "Found Zstd via pkg-config: ${ZSTD_LIBRARIES}")
    add_library(zstd_target ALIAS PkgConfig::ZSTD)
endif()

if(TORCH_CUDA_ARCH_LIST OR CUDA_FOUND OR HIP_FOUND)
    set(USE_CUDA ON)
    message(STATUS "GPU Support Enabled (Torch/CUDA/HIP detected)")
    
    if(HIP_FOUND)
        message(STATUS ">>> ROCm (HIP) Environment Detected")
    endif()
else()
    set(USE_CUDA OFF)
    message(STATUS "GPU Support Disabled (Using CPU Only)")
endif()

set(NVCOMP_FOUND OFF)

if(USE_CUDA)
    message(STATUS "Checking for nvcomp (NVIDIA specific)...")
    
    find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.h nvcomp/nvcomp.h)
    find_library(NVCOMP_LIBRARY NAMES nvcomp)
    
    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
        set(NVCOMP_FOUND TRUE)
        message(STATUS ">>> Found nvcomp: ${NVCOMP_LIBRARY}")
    else()
        message(STATUS ">>> nvcomp NOT found. Will use Hybrid Mode (GPU PCA + CPU ZSTD)")
    endif()
endif()

set(CAESAR_SRC
    CAESAR.cpp
    dataset/dataset.cpp
    models/range_coder/rans_coder.cpp
    models/caesar_compress.cpp
    models/caesar_decompress.cpp
    models/model_utils.cpp
    models/runGaeCuda.cpp  
    data_utils.cpp
)

add_library(caesar_lib SHARED ${CAESAR_SRC})

target_compile_definitions(caesar_lib PRIVATE
    DEFAULT_CAESAR_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/caesar/models"
)

if(USE_CUDA)
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    message(STATUS "Build defined with USE_CUDA")
    
    if(HIP_FOUND)
        target_compile_definitions(caesar_lib PUBLIC USE_ROCM)
    endif()
endif()

if(NVCOMP_FOUND)
    target_compile_definitions(caesar_lib PRIVATE ENABLE_NVCOMP)
    message(STATUS "Build defined with ENABLE_NVCOMP")
endif()

target_include_directories(caesar_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/range_coder>
        $<INSTALL_INTERFACE:include/caesar>
)

target_link_libraries(caesar_lib PUBLIC zstd_target torch)

if(USE_CUDA)
    if(CUDA_FOUND)
        target_include_directories(caesar_lib PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES})
    endif()
    
    if(NVCOMP_FOUND)
        target_include_directories(caesar_lib PRIVATE ${NVCOMP_INCLUDE_DIR})
        target_link_libraries(caesar_lib PUBLIC ${NVCOMP_LIBRARY})
    endif()
endif()

if(MPI_FOUND)
    target_include_directories(caesar_lib PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(caesar_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(caesar_lib PUBLIC USE_MPI)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(caesar_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(caesar_lib PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native -fprefetch-loop-arrays>
    $<$<CONFIG:Profile>:-O3 -g -fno-omit-frame-pointer -fno-inline-functions>
)

if(USE_CUDA)
    target_compile_options(caesar_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3 --maxrregcount=64>
    )
endif()

set_target_properties(caesar_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)