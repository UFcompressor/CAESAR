# Add subdirectory for BCRN
add_subdirectory(BCRN)

# Add compressModules3dMidSR library
add_library(compressModules3dMidSR
    compressModules3dMidSR.cpp
    compressModules3dMidSR.h
)

target_include_directories(compressModules3dMidSR
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(compressModules3dMidSR
    PUBLIC "${TORCH_LIBRARIES}"
    PUBLIC caesar_lib
)


add_library(utils
    utils.cpp
    utils.h
)

target_include_directories(utils
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(utils
    PUBLIC "${TORCH_LIBRARIES}"
    PUBLIC caesar_lib
)

