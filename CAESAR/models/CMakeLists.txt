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

# Add utils library
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

# Set prefix path to your local nvCOMP install
set(CMAKE_PREFIX_PATH "$ENV{HOME}/local/nvcomp" ${CMAKE_PREFIX_PATH})

# Find nvCOMP
find_package(nvcomp CONFIG REQUIRED)


add_library(runGaeCuda
    runGaeCuda.cpp
    runGaeCuda.h
)

target_include_directories(runGaeCuda
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PUBLIC "${CMAKE_SOURCE_DIR}/external"
)

target_link_libraries(runGaeCuda
    PUBLIC "${TORCH_LIBRARIES}"
    PUBLIC caesar_lib
    PUBLIC nlohmann_json
    PUBLIC nvcomp::nvcomp    
)

