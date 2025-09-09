cmake_minimum_required(VERSION 3.18)
project(caesar)

# cuda is required  but figure out how to make it not require with others
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)


add_subdirectory(CAESAR)
add_subdirectory(tests)

