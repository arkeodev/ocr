#!/bin/bash
# Simple script to run the OCR app in terminal-only mode

# Ensure we exit on errors
set -e

# Show colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}OCR 2.0 Terminal Runner${NC}"
echo "------------------------"

# Parse arguments
CPU_ONLY=false
DEBUG_LEVEL=2  # Default to INFO level

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu|-c)
            CPU_ONLY=true
            shift
            ;;
        --download|-d)
            echo -e "${GREEN}Downloading model to cache...${NC}"
            poetry run download-model
            exit 0
            ;;
        --debug|-v)
            DEBUG_LEVEL=3  # DEBUG level
            shift
            ;;
        --quiet|-q)
            DEBUG_LEVEL=1  # WARNING level
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_terminal.sh [options]"
            echo "Options:"
            echo "  --cpu, -c         Force CPU mode (no GPU)"
            echo "  --download, -d    Download model to cache and exit"
            echo "  --debug, -v       Enable verbose debug output"
            echo "  --quiet, -q       Show only warnings and errors"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Poetry is not installed. Installing Poetry...${NC}"
    pip install poetry
fi

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
poetry install > /dev/null 2>&1

# Check if we need to install specific packages that might cause issues
echo -e "${GREEN}Ensuring all critical dependencies are installed...${NC}"
poetry run pip install --upgrade torch torchvision accelerate > /dev/null 2>&1

# Apple Silicon optimization
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    echo -e "${YELLOW}Detected Apple Silicon (M1/M2/M3) Mac${NC}"
    echo -e "${GREEN}Setting optimized environment variables for MPS...${NC}"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

# Set environment variables based on options
if [ "$CPU_ONLY" = true ]; then
    echo -e "${YELLOW}Forcing CPU mode as requested${NC}"
    export FORCE_CPU=1
fi

# Set debug level
export DEBUG_LEVEL=$DEBUG_LEVEL

# Run the app
echo -e "${GREEN}Starting OCR processing...${NC}"
echo -e "${YELLOW}If you encounter any issues, try running:${NC}"
echo -e "${YELLOW}./run_terminal.sh --cpu${NC}"
echo "------------------------"

# Run with Python directly
poetry run python src/ocr_app/app.py 