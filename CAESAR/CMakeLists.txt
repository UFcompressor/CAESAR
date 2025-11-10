find_package(MPI)
find_package(CUDA)
find_package(PkgConfig)

# Zstd
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
endif()

if(NOT ZSTD_FOUND)
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY NAMES zstd)
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        set(ZSTD_FOUND TRUE)
        message(STATUS "Found Zstd: ${ZSTD_LIBRARY}")
    else()
        message(FATAL_ERROR "Zstd library not found. Please install libzstd-dev")
    endif()
endif()

# CUDA + nvcomp
set(CUDA_SUPPORT_ENABLED FALSE)
if(CUDA_FOUND)
    message(STATUS "CUDA found, checking for nvcomp...")

    find_path(NVCOMP_INCLUDE_DIR
        NAMES nvcomp.h nvcomp/nvcomp.h
        HINTS ENV NVCOMP_ROOT ENV NVCOMP_DIR
        PATH_SUFFIXES include
    )

    find_library(NVCOMP_LIBRARY
        NAMES nvcomp
        HINTS ENV NVCOMP_ROOT ENV NVCOMP_DIR
        PATH_SUFFIXES lib lib64
    )

    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
        set(NVCOMP_FOUND TRUE)
        message(STATUS "Found nvcomp: ${NVCOMP_LIBRARY}")
        message(STATUS "nvcomp include dir: ${NVCOMP_INCLUDE_DIR}")
        set(CUDA_SUPPORT_ENABLED TRUE)
    else()
        message(WARNING "CUDA found but nvcomp NOT found. Building WITHOUT CUDA support.")
    endif()
else()
    message(STATUS "CUDA not found, building without CUDA support")
endif()

# Library
add_library(caesar_lib SHARED
    CAESAR.cpp
    dataset/dataset.cpp
    models/runGaeCuda.cpp
    models/range_coder/rans_coder.cpp
    models/caesar_compress.cpp
    models/caesar_decompress.cpp
)

# Include directories
target_include_directories(caesar_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/range_coder>
        $<INSTALL_INTERFACE:include/caesar>
)

# Torch
target_link_libraries(caesar_lib PUBLIC ${TORCH_LIBRARIES})

# CUDA support
if(CUDA_SUPPORT_ENABLED)
    message(STATUS "Enabling CUDA support with nvcomp")
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    target_include_directories(caesar_lib PRIVATE ${CUDA_INCLUDE_DIRS} ${NVCOMP_INCLUDE_DIR})
    target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES} ${NVCOMP_LIBRARY})
endif()

# Zstd support
if(ZSTD_FOUND)
    if(TARGET PkgConfig::ZSTD)
        target_link_libraries(caesar_lib PUBLIC PkgConfig::ZSTD)
    else()
        target_include_directories(caesar_lib PRIVATE ${ZSTD_INCLUDE_DIR})
        target_link_libraries(caesar_lib PUBLIC ${ZSTD_LIBRARY})
    endif()
    message(STATUS "Zstd support enabled")
endif()

# MPI support
if(MPI_FOUND)
    target_include_directories(caesar_lib PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(caesar_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(caesar_lib PUBLIC USE_MPI)
endif()

# Set Debug/Release compiler flags
target_compile_options(caesar_lib PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
)

# Versioning
set_target_properties(caesar_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    EXPORT_NAME caesar_lib
)
