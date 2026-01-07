
if(APPLE)
    set(FILESYSTEM_LIB "")
elseif(UNIX AND NOT APPLE)
    set(FILESYSTEM_LIB "stdc++fs")
else()
    set(FILESYSTEM_LIB "")
endif()

if(TORCH_CUDA_ARCH_LIST OR CUDA_FOUND OR HIP_FOUND)
    set(USE_CUDA ON)
    message(STATUS "Tests: GPU Support Enabled")
else()
    set(USE_CUDA OFF)
    message(STATUS "Tests: GPU Support Disabled (CPU Only)")
endif()

set(NVCOMP_FOUND OFF)
if(USE_CUDA)
    find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.h nvcomp/nvcomp.h)
    find_library(NVCOMP_LIBRARY NAMES nvcomp)

    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
        set(NVCOMP_FOUND TRUE)
        message(STATUS "Tests: CUDA + nvcomp found")
    else()
        message(STATUS "Tests: nvcomp NOT found (Hybrid/AMD Mode)")
    endif()
endif()

add_executable(test_dataset testDataSet.cpp)
target_link_libraries(test_dataset PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_caesarCD test_caesarCD.cpp)
target_link_libraries(test_caesarCD PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_rans test_range_coder_consistency.cpp)
target_link_libraries(test_rans PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_caesarC test_caesarC.cpp)
target_link_libraries(test_caesarC PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_caesarD test_decompress_consistency.cpp)
target_link_libraries(test_caesarD PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_datasetNRMSE testDatasetNRMSE.cpp)
target_link_libraries(test_datasetNRMSE PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda PRIVATE caesar_lib "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

if(USE_CUDA)
    target_compile_definitions(test_runGaeCuda PRIVATE USE_CUDA)
    
    if(NVCOMP_FOUND)
        target_compile_definitions(test_runGaeCuda PRIVATE ENABLE_NVCOMP)
        target_include_directories(test_runGaeCuda PRIVATE ${NVCOMP_INCLUDE_DIR})
    endif()

    if(HIP_FOUND)
        target_compile_definitions(test_runGaeCuda PRIVATE USE_ROCM)
    endif()

    if(CUDA_FOUND)
        target_include_directories(test_runGaeCuda PRIVATE ${CUDA_INCLUDE_DIRS})
    endif()

    message(STATUS "Building test_runGaeCuda with GPU support")
else()
    message(STATUS "Building test_runGaeCuda in CPU-only mode")
endif()

set(TEST_TARGETS
    test_dataset
    test_caesarC
    test_caesarCD
    test_datasetNRMSE
    test_rans
    test_caesarD
    test_runGaeCuda
)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(test_exe ${TEST_TARGETS})
        add_custom_command(TARGET ${test_exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:${test_exe}>)
    endforeach()
endif()
