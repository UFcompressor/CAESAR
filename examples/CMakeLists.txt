add_executable(hello_caesar hello_caesar.cpp)

target_link_libraries(hello_caesar
    PRIVATE
        caesar
        "${TORCH_LIBRARIES}"
)

set_target_properties(hello_caesar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)
