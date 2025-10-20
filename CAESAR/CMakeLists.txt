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

add_library(caesar_lib SHARED
    CAESAR.cpp
    normalize.cpp
    dataset/dataset.cpp
    pretrainedModelIO/CaesarModelLoader.cpp
    models/compressModules3dMidSR.cpp
    models/utils.cpp
    models/networkComponents.cpp
    models/BCRN/blocks.cpp
    models/BCRN/bcrnModel.cpp
    models/compress/compressor.cpp
    models/runGaeCuda.cpp
)

target_include_directories(caesar_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pretrainedModelIO>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/BCRN>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/compress>
        $<INSTALL_INTERFACE:include/caesar>
)

target_link_libraries(caesar_lib PUBLIC ${TORCH_LIBRARIES})

if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling CUDA support in runGaeCuda.cpp")
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    target_include_directories(caesar_lib PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA not found, building runGaeCuda.cpp without CUDA")
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

set_target_properties(caesar_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    EXPORT_NAME caesar_lib
)
