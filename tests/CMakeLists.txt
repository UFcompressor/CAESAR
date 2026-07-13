find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

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


add_executable(test_caesarCD test_caesarCD.cpp)
target_link_libraries(test_caesarCD PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})



add_executable(test_caesarC test_caesarC.cpp)
target_link_libraries(test_caesarC PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})


add_executable(test_caesarD test_caesarD.cpp)
target_link_libraries(test_caesarD PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})


add_executable(test_padding test_padding.cpp)
target_link_libraries(test_padding PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_LBRC testLBRC.cpp)
target_link_libraries(test_LBRC PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

add_executable(test_NGLR test_NGLR.cpp)
target_link_libraries(test_NGLR PRIVATE caesar "${TORCH_LIBRARIES}" ${FILESYSTEM_LIB})

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
    test_caesarCD
    test_caesarC
    test_caesarD
    test_padding
    test_runGaeCuda
    test_LBRC
    test_NGLR
)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(test_exe ${TEST_TARGETS})
        add_custom_command(TARGET ${test_exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:${test_exe}>)
    endforeach()
endif()

foreach(t IN ITEMS test_caesarCD test_caesarC test_caesarD test_padding test_LBRC test_runGaeCuda test_NGLR)
    target_compile_options(${t} PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fPIC>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -mtune=native -fprefetch-loop-arrays>
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-g -O0>
    )
    target_compile_definitions(${t} PRIVATE
        DEFAULT_CAESAR_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/caesar/models"
        MODEL_CAESAR_V_ONLY
    )
    target_precompile_headers(${t} REUSE_FROM caesar_objs)
endforeach()
