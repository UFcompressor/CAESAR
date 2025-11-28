cmake_minimum_required(VERSION 3.18)
project(CAESAR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type (Debug, Release, or Profile)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS ">> Debug mode enabled")
    add_compile_definitions(DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS ">> Release mode enabled")
    add_compile_definitions(NDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    message(STATUS ">> Profile mode enabled")
    add_compile_definitions(NDEBUG PROFILE_MODE)
else()
    message(WARNING "Unknown build type: ${CMAKE_BUILD_TYPE}")
    message(FATAL_ERROR "Please Choose a Build Type: Ex: Release, Debug, Profile")
endif()

find_package(Torch REQUIRED)

add_subdirectory(CAESAR)

option(BUILD_TESTS "Build CAESAR test files" OFF)

if(BUILD_TESTS)
    message(STATUS "Building test files...")
    add_subdirectory(tests)
else()
    message(STATUS "Skipping test files (BUILD_TESTS=OFF)")
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CAESAR/
        DESTINATION include/caesar
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/exported_model/
        DESTINATION share/caesar/models
        FILES_MATCHING PATTERN "*.bin" PATTERN "*.pt2")
        
add_library(caesar::caesar ALIAS caesar_lib)

install(TARGETS caesar_lib
  EXPORT CAESARTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(EXPORT CAESARTargets
  FILE CAESARTargets.cmake
  NAMESPACE caesar::
  DESTINATION lib/cmake/caesar)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CAESARConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CAESARConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CAESARConfig.cmake"
  INSTALL_DESTINATION lib/cmake/caesar)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/CAESARConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CAESARConfigVersion.cmake"
  DESTINATION lib/cmake/caesar)

