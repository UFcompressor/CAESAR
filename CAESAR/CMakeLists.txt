find_package(MPI)
find_package(CUDA)
find_package(PkgConfig)


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

set(USE_CUDA OFF)

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
        set(USE_CUDA ON)
    else()
        message(WARNING "CUDA found but nvcomp NOT found → CUDA DISABLED")
    endif()
else()
    message(STATUS "CUDA NOT found → CPU ONLY BUILD")
endif()

set(CAESAR_SRC
    CAESAR.cpp
    dataset/dataset.cpp
    models/range_coder/rans_coder.cpp
    models/caesar_compress.cpp
    models/caesar_decompress.cpp
    models/model_utils.cpp
)


if(USE_CUDA)
    list(APPEND CAESAR_SRC models/runGaeCuda.cpp)
endif()

add_library(caesar_lib SHARED ${CAESAR_SRC})

target_compile_definitions(caesar_lib PRIVATE
    DEFAULT_CAESAR_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/caesar/models"
)


if(USE_CUDA)
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    message(STATUS "USE_CUDA ENABLED")
else()
    message(STATUS "USE_CUDA DISABLED")
endif()

target_include_directories(caesar_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/range_coder>
        $<INSTALL_INTERFACE:include/caesar>
)

target_link_libraries(caesar_lib PUBLIC ${TORCH_LIBRARIES})

if(USE_CUDA)
    message(STATUS "Enabling CUDA support (CUDA + nvcomp detected)")
    target_include_directories(caesar_lib PRIVATE ${CUDA_INCLUDE_DIRS} ${NVCOMP_INCLUDE_DIR})
    target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES} ${NVCOMP_LIBRARY})
endif()

if(ZSTD_FOUND)
    if(TARGET PkgConfig::ZSTD)
        target_link_libraries(caesar_lib PUBLIC PkgConfig::ZSTD)
    else()
        target_include_directories(caesar_lib PRIVATE ${ZSTD_INCLUDE_DIR})
        target_link_libraries(caesar_lib PUBLIC ${ZSTD_LIBRARY})
    endif()
    message(STATUS "Zstd support enabled")
endif()

if(MPI_FOUND)
    target_include_directories(caesar_lib PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(caesar_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(caesar_lib PUBLIC USE_MPI)
endif()

target_compile_options(caesar_lib PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Profile>:-O3 -g>
)

set_target_properties(caesar_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)
