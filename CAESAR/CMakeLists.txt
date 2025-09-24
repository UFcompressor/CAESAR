# Try to find MPI, but don’t require it
find_package(MPI)

add_library(caesar_lib
    compressor.cpp
    normalize.cpp

    # Dataset
    dataset/dataset.cpp

    # Pretrained model IO
    pretrainedModelIO/CaesarModelLoader.cpp

    # Models
    models/compressModules3dMidSR.cpp
    models/utils.cpp
    models/networkComponents.cpp
    models/runGaeCuda.cpp

    # BCRN
    models/BCRN/blocks.cpp
    models/BCRN/bcrnModel.cpp
)

target_include_directories(caesar_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/dataset
        ${CMAKE_CURRENT_SOURCE_DIR}/pretrainedModelIO
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/models/BCRN
)

target_link_libraries(caesar_lib
    PUBLIC
        "${TORCH_LIBRARIES}"
)

# If MPI was found, add its includes, link libs, and a flag
if(MPI_FOUND)
    target_include_directories(caesar_lib PUBLIC ${MPI_INCLUDE_PATH})
    target_link_libraries(caesar_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(caesar_lib PUBLIC USE_MPI)
endif()

