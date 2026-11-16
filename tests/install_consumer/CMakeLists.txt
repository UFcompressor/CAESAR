cmake_minimum_required(VERSION 3.18)
project(CAESARInstallTest LANGUAGES CXX)

find_package(CAESAR CONFIG REQUIRED)
add_executable(install_consumer main.cpp)
target_compile_features(install_consumer PRIVATE cxx_std_17)
target_link_libraries(install_consumer PRIVATE caesar::caesar)
