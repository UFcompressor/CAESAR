# Test: Caesar Reader
add_executable(test_caesar testCaesarReader.cpp)
target_link_libraries(test_caesar
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: BSConvU
add_executable(test_bsconvu testBsconvu.cpp)
target_link_libraries(test_bsconvu
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: Blocks
add_executable(test_blocks testBlocks.cpp)
target_link_libraries(test_blocks
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: BCRN Model
add_executable(test_bcrn testBcrn.cpp)
target_link_libraries(test_bcrn
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: Utils
add_executable(test_utils testUtils.cpp)
target_link_libraries(test_utils
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

add_executable(test_utils2 testUtils2.cpp)
target_link_libraries(test_utils2
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: runGaeCuda
add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: networkComponents
add_executable(test_networkComponents testNetworkComponents.cpp)
target_link_libraries(test_networkComponents
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

add_executable(test_dataset testDataSet.cpp)
target_link_libraries(test_dataset
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)

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
)

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(test_exe ${TEST_TARGETS})
    add_custom_command(TARGET ${test_exe}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:${test_exe}>)
  endforeach()
endif()

