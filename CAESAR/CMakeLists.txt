add_library(caesar_lib
    compressor.cpp
    normalize.cpp
)

# Include current dir + pretrainedModelReader
target_include_directories(caesar_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/pretrainedModelIO
)

# Link Torch
target_link_libraries(caesar_lib
    PUBLIC
        "${TORCH_LIBRARIES}"
)

# Add subdirectories
add_subdirectory(pretrainedModelIO)
add_subdirectory(models)  
