# this is for parallel IO for later when using CARSAR directly
find_package(MPI)
find_package(CUDA) 

add_library(caesar_lib
    CAESAR.cpp
    normalize.cpp

    # Dataset
    dataset/dataset.cpp

    # Pretrained model IO
    pretrainedModelIO/CaesarModelLoader.cpp

    # Models

    # will go away
    # --------------------------------------------
    models/compressModules3dMidSR.cpp
    models/utils.cpp
    models/networkComponents.cpp
    # --------------------------------------------

    # BCRN
    # will go away
    # ----------------------------
    models/BCRN/blocks.cpp
    models/BCRN/bcrnModel.cpp
    # -----------------------------------

    models/compress/compressor.cpp
)

# Add runGaeCuda.cpp only if CUDA is available
if(CUDA_FOUND)
    message(STATUS "CUDA found, building with CUDA support")
    target_sources(caesar_lib PRIVATE models/runGaeCuda.cpp)
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    target_include_directories(caesar_lib PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA not found, building without CUDA support")
endif()

target_include_directories(caesar_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/dataset
        ${CMAKE_CURRENT_SOURCE_DIR}/pretrainedModelIO
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/models/BCRN
        ${CMAKE_CURRENT_SOURCE_DIR}/models/compress
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

