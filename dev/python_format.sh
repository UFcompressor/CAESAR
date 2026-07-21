#!/bin/bash
set -e

usage() {
    echo "Usage: $(basename"$0") [FILE_OR_DIR]..."
    echo
    echo "Format Python files using Black."
    echo
    echo "Examples:"
    echo "  $(basename "$0")              # Format current directory"
    echo "  $(basename "$0") tests        # Format a directory"
    echo "  $(basename "$0") file.py      # Format a single file"
    echo "  $(basename "$0") src tests    # Format multiple paths"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v black >/dev/null 2>&1; then
    echo "Error: black is not installed."
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

    echo "Formatting $path..."
    black "$path"
done 
