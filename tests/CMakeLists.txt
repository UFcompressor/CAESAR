find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found, enabling MPI tests")
    set(MPI_LIB MPI::MPI_CXX)
else()
    message(STATUS "MPI not found, skipping MPI linking")
    set(MPI_LIB "")
endif()


# Test: runGaeCuda
add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: DataSet
add_executable(test_dataset testDataSet.cpp)
target_link_libraries(test_dataset PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: DataSet2 (MPI-enabled)
add_executable(test_dataset2 testDataSet2.cpp)
# Only link if MPI was found
target_link_libraries(test_dataset2 PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs ${MPI_LIB})


# Step 1: Declare the executable with only source files
add_executable(test_caesarCD test_caesarCD.cpp)

# Step 2: Link libraries with PRIVATE keyword
target_link_libraries(test_caesarCD
    PRIVATE
    caesar_lib
    "${TORCH_LIBRARIES}"
    stdc++fs
)

set(TEST_TARGETS
    test_runGaeCuda
    test_dataset
    test_dataset2
    test_caesarCD
)

# Copy DLLs on MSVC if needed
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(test_exe ${TEST_TARGETS})
        add_custom_command(TARGET ${test_exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:${test_exe}>)
    endforeach()
endif()

