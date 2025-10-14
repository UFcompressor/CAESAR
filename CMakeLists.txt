cmake_minimum_required(VERSION 3.18)
project(caesar)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_subdirectory(CAESAR)
add_subdirectory(tests)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CAESAR/
        DESTINATION include/caesar
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

add_library(caesar::caesar ALIAS caesar_lib)

install(TARGETS caesar_lib
  EXPORT caesarTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(EXPORT caesarTargets
  FILE caesarTargets.cmake
  NAMESPACE caesar::
  DESTINATION lib/cmake/caesar)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/caesarConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/caesarConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/caesarConfig.cmake"
  INSTALL_DESTINATION lib/cmake/caesar)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/caesarConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/caesarConfigVersion.cmake"
  DESTINATION lib/cmake/caesar)
