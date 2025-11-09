find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found, enabling MPI tests")
    set(MPI_LIB MPI::MPI_CXX)
else()
    message(STATUS "MPI not found, skipping MPI linking")
    set(MPI_LIB "")
endif()



add_executable(test_runGaeCuda testRunGaeCuda.cpp)
target_link_libraries(test_runGaeCuda PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )


add_executable(test_dataset testDataSet.cpp)
target_link_libraries(test_dataset PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

add_executable(test_dataset2 testDataSet2.cpp)
target_link_libraries(test_dataset2 PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs ${MPI_LIB})



#add_executable(test_caesarCD test_caesarCD.cpp)
#target_link_libraries(test_caesarCD
#    PRIVATE
#    caesar_lib
#    "${TORCH_LIBRARIES}"
#    stdc++fs
#)

add_executable(test_rans test_range_coder_consistency.cpp)
target_link_libraries(test_rans PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

add_executable(test_caesarC test_caesarC.cpp)
target_link_libraries(test_caesarC
    PRIVATE
    caesar_lib
    "${TORCH_LIBRARIES}"
    stdc++fs
)

add_executable(test_caesarD test_decompress_consistency.cpp)
target_link_libraries(test_caesarD PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

add_executable(test_datasetNRMSE testDatasetNRMSE.cpp)
target_link_libraries(test_datasetNRMSE PRIVATE caesar_lib "${TORCH_LIBRARIES}" stdc++fs )

set(TEST_TARGETS
    test_runGaeCuda
    test_dataset
    test_dataset2
    test_caesarC
    test_caesarCD
    test_datasetNRMSE
    test_rans
    test_caesarD
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

