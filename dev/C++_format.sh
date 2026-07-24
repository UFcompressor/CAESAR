#!/bin/bash
set -e

usage() {
    echo "Usage: $(basename "$0") [FILE_OR_DIR]..."
    echo
    echo "Format C/C++ source and header files using clang-format."
    echo
    echo "Supported extensions:"
    echo "  .cpp .cc .cxx .c .hpp .hh .hxx .h"
    echo
    echo "Examples:"
    echo "  $(basename "$0")              # Format current directory"
    echo "  $(basename "$0") src          # Format a directory"
    echo "  $(basename "$0") file.cpp     # Format a single file"
    echo "  $(basename "$0") src include  # Format multiple paths"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v clang-format >/dev/null 2>&1; then
    echo "Error: clang-format is not installed."
    exit 1
fi

if [ $# -eq 0 ]; then
    set -- .
fi

for path in "$@"; do
    if [ ! -e "$path" ]; then
        echo "Warning: '$path' does not exist. Skipping."
        continue
    fi

    if [ -f "$path" ]; then
        case "$path" in
            *.cpp|*.cc|*.cxx|*.c|*.hpp|*.hh|*.hxx|*.h)
                echo "Formatting $path..."
                clang-format -i "$path"
                ;;
            *)
                echo "Skipping '$path' (not a supported C/C++ file)."
                ;;
        esac
    else
        echo "Formatting $path..."
        find "$path" \
            \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" \
               -o -name "*.hpp" -o -name "*.hh" -o -name "*.hxx" -o -name "*.h" \) \
            -exec clang-format -i {} +
    fi
done
