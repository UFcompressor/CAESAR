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
    PUBLIC "${TORCH_LIBRARIES}"
)

