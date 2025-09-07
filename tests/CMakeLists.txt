# Test: Caesar Reader
add_executable(test_caesar
    testCaesarReader.cpp
)
target_link_libraries(test_caesar
    PRIVATE
        caesar_lib
        compressModules3dMidSR
        BCRN
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: BSConvU
add_executable(test_bsconvu
    testBsconvu.cpp
)
target_link_libraries(test_bsconvu
    PRIVATE
        caesar_lib
        BCRN
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: Blocks
add_executable(test_blocks
    testBlocks.cpp
)
target_link_libraries(test_blocks
    PRIVATE
        caesar_lib
        BCRN
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: BCRN Model
add_executable(test_bcrn
    testBcrn.cpp
)
target_link_libraries(test_bcrn
    PRIVATE
        BCRN
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Test: Utils
add_executable(test_utils
    testUtils.cpp
)

# LINK the utils library AND make sure CAESAR/models is added before tests
target_link_libraries(test_utils
    PRIVATE
        caesar_lib
        utils        # link the utils library
        "${TORCH_LIBRARIES}"
        stdc++fs
)

add_executable(test_utils2
    testUtils2.cpp
)

target_link_libraries(test_utils2
    PRIVATE
        caesar_lib
        utils        # link the utils library
        "${TORCH_LIBRARIES}"
        stdc++fs
)

# Collect all tests
set(TEST_TARGETS
    test_caesar
    test_bsconvu
    test_blocks
    test_bcrn
    test_utils
    test_utils2
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

