# Find MPI (optional)
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found, enabling MPI tests")
    set(MPI_LIB MPI::MPI_CXX)
else()
    message(STATUS "MPI not found, skipping MPI linking")
    set(MPI_LIB "")
endif()

# Test: Caesar Reader
add_executable(test_caesar testCaesarReader.cpp)
target_link_libraries(test_caesar PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: BSConvU
add_executable(test_bsconvu testBsconvu.cpp)
target_link_libraries(test_bsconvu PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: Blocks
add_executable(test_blocks testBlocks.cpp)
target_link_libraries(test_blocks PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: BCRN Model
add_executable(test_bcrn testBcrn.cpp)
target_link_libraries(test_bcrn PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: Utils
add_executable(test_utils testUtils.cpp)
target_link_libraries(test_utils PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )
add_executable(test_utils2 testUtils2.cpp)
target_link_libraries(test_utils2 PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: runGaeCuda
add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: networkComponents
add_executable(test_networkComponents testNetworkComponents.cpp)
target_link_libraries(test_networkComponents PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: DataSet
add_executable(test_dataset testDataSet.cpp)
target_link_libraries(test_dataset PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

# Test: DataSet2 (MPI-enabled)
add_executable(test_dataset2 testDataSet2.cpp)
# Only link if MPI was found
target_link_libraries(test_dataset2 PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs ${MPI_LIB})

#Test: Comrpess
add_executable(test_compress testCompress.cpp)
target_link_libraries(test_compress PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs)

# Collect all targets
set(TEST_TARGETS
    test_caesar
    test_bsconvu
    test_blocks
    test_bcrn
    test_utils
    test_utils2
    test_runGaeCuda
    test_networkComponents
    test_dataset
    test_dataset2
    test_compress
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

