find_package(CUDA)
find_package(PkgConfig)
find_package(Torch REQUIRED)

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
    models/lbrc.cpp
    data_utils.cpp
)

if(MSVC)
    add_library(caesar STATIC ${CAESAR_SRC})
else()
    add_library(caesar SHARED ${CAESAR_SRC})
endif()

target_compile_definitions(caesar PRIVATE
    DEFAULT_CAESAR_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/caesar/models"
)

if(USE_CUDA)
    target_compile_definitions(caesar PUBLIC USE_CUDA)
    message(STATUS "Build defined with USE_CUDA")

    if(HIP_FOUND)
        target_compile_definitions(caesar PUBLIC USE_ROCM)
    endif()
endif()

if(NVCOMP_FOUND)
    target_compile_definitions(caesar PRIVATE ENABLE_NVCOMP)
    message(STATUS "Build defined with ENABLE_NVCOMP")
endif()

# todo: fix this to read in the file and set it to just some model name
target_compile_definitions(caesar PRIVATE MODEL_CAESAR_V_ONLY)
message(STATUS "Build configured for CAESAR-V only ")

target_include_directories(caesar
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/range_coder>
        $<INSTALL_INTERFACE:include/caesar>
)

target_link_libraries(caesar PUBLIC zstd_target torch)

if(USE_CUDA)
    if(CUDA_FOUND)
        target_include_directories(caesar PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(caesar PUBLIC ${CUDA_LIBRARIES})
    endif()

    if(NVCOMP_FOUND)
        target_include_directories(caesar PRIVATE ${NVCOMP_INCLUDE_DIR})
        target_link_libraries(caesar PUBLIC ${NVCOMP_LIBRARY})
    endif()
endif()

target_compile_options(caesar PRIVATE
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-g -O0>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -mtune=native -fprefetch-loop-arrays>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Profile>>:-O3 -g -fno-omit-frame-pointer -fno-inline-functions>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Zi /Od /wd4267>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /wd4267>
)

if(USE_CUDA)
    target_compile_options(caesar PRIVATE
        $<$<AND:$<CONFIG:Profile>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
    )

    target_compile_options(caesar PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3 --maxrregcount=64>
    )
endif()

set_target_properties(caesar PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

add_executable(caesar_cli CAESAR.cpp)

target_link_libraries(caesar_cli PRIVATE caesar torch)

if(USE_CUDA)
    target_compile_definitions(caesar_cli PRIVATE USE_CUDA)
    if(HIP_FOUND)
        target_compile_definitions(caesar_cli PRIVATE USE_ROCM)
    endif()
endif()

target_compile_definitions(caesar_cli PRIVATE MODEL_CAESAR_V_ONLY)

target_compile_options(caesar_cli PRIVATE
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-g -O0>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -mtune=native>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Profile>>:-O3 -g -fno-omit-frame-pointer>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Zi /Od /wd4267>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /wd4267>
)

if(MSVC)
    set_target_properties(caesar_cli PROPERTIES OUTPUT_NAME caesar_cli)
else()
    set_target_properties(caesar_cli PROPERTIES OUTPUT_NAME caesar)
endif()

install(TARGETS caesar
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS caesar_cli
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/caesar
    FILES_MATCHING PATTERN "*.h"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/pretrained")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models/pretrained/
        DESTINATION share/caesar/models
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../exported_model/model_name.txt")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../exported_model/model_name.txt"
        DESTINATION share/caesar/models
    )
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../exported_model/model_name.txt" CAESAR_MODEL_NAME)
    string(STRIP "${CAESAR_MODEL_NAME}" CAESAR_MODEL_NAME)
else()
    message(FATAL_ERROR "model_name.txt not found in exported_model/. Run the export script first.")
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "CAESAR Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "GPU Support:       ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "nvcomp Support:    ${NVCOMP_FOUND}")
endif()
message(STATUS "Model Name:        ${CAESAR_MODEL_NAME}")
message(STATUS "CLI Executable:    caesar")
message(STATUS "Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
