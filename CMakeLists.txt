cmake_minimum_required(VERSION 3.18)
project(caesar)

# cuda is required  but figure out how to make it not require with others
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_SOURCE_DIR}/external/nlohmann
)


add_subdirectory(CAESAR)
add_subdirectory(tests)

