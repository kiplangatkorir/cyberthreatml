#!/bin/bash
#
# Script to run Python examples in the CyberThreat-ML project
# Usage: bash run_python_example.sh <example_path>
# Example: bash run_python_example.sh examples/minimal_example.py

# Python path in Replit environment
PYTHON_PATH="/mnt/nixmodules/nix/store/b03kwd9a5dm53k0z5vfzdhkvaa64c4g7-python3-3.10.13-env/bin/python3"

# Check if the Python path exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python interpreter not found at $PYTHON_PATH"
    echo "Trying to locate Python..."
    
    # Try to find another Python interpreter
    ALT_PYTHON=$(which python3 2>/dev/null)
    if [ -z "$ALT_PYTHON" ]; then
        ALT_PYTHON=$(which python 2>/dev/null)
    fi
    
    if [ -n "$ALT_PYTHON" ]; then
        echo "Found alternative Python at $ALT_PYTHON"
        PYTHON_PATH=$ALT_PYTHON
    else
        echo "Error: No Python interpreter found. Please install Python 3."
        exit 1
    fi
fi

# Check if an example was specified
if [ -z "$1" ]; then
    echo "Error: No example specified."
    echo "Usage: bash run_python_example.sh <example_path>"
    echo "Example: bash run_python_example.sh examples/minimal_example.py"
    exit 1
fi

# Check if the example file exists
if [ ! -f "$1" ]; then
    echo "Error: Example file '$1' not found."
    echo "Available examples:"
    find examples -name "*.py" | sort
    exit 1
fi

# Set the working directory to the project root
cd "$(dirname "$0")"

# Run the example
echo "Running $1 with Python $PYTHON_PATH..."
echo "------------------------------------"
$PYTHON_PATH "$1"

# Exit with the status of the Python command
exit $?