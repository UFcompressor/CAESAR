
add_subdirectory(BCRN)


# compressModules3dMidSR library
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


# utils library
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


# networkComponents library
add_library(networkComponents
    networkComponents.cpp
    networkComponents.h
)

target_include_directories(networkComponents
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(networkComponents
    PUBLIC "${TORCH_LIBRARIES}"
    PUBLIC caesar_lib
)


# Optional: nvCOMP
if(DEFINED ENV{NVCOMP_DIR})
    list(APPEND CMAKE_PREFIX_PATH $ENV{NVCOMP_DIR})
endif()

find_package(nvcomp CONFIG QUIET)

if(nvcomp_FOUND)
    message(STATUS "nvCOMP found, enabling GPU compression support")
else()
    message(STATUS "nvCOMP not found, building without GPU compression support")
endif()


# runGaeCuda library
add_library(runGaeCuda
    runGaeCuda.cpp
    runGaeCuda.h
)

target_include_directories(runGaeCuda
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(runGaeCuda
    PUBLIC "${TORCH_LIBRARIES}"
    PUBLIC caesar_lib
)

# Link nvCOMP only if available
if(nvcomp_FOUND)
    target_link_libraries(runGaeCuda PUBLIC nvcomp::nvcomp)
endif()

