#!/bin/bash
# Simple script to run the OCR app with Poetry

# Ensure we exit on errors
set -e

# Show colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}OCR 2.0 Demo App${NC}"
echo "------------------------"

# Parse arguments
CPU_ONLY=false
DEBUG_LEVEL=2  # Default to INFO level
SHOW_DEBUG_UI=0
CLEAN_OUTPUT=false
TERMINAL_ONLY=0

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
            SHOW_DEBUG_UI=1
            shift
            ;;
        --quiet|-q)
            DEBUG_LEVEL=1  # WARNING level
            shift
            ;;
        --clean|-cl)
            CLEAN_OUTPUT=true
            shift
            ;;
        --terminal-only|-t)
            TERMINAL_ONLY=1
            echo -e "${YELLOW}Terminal-only mode: all logs will go to console, no Streamlit UI logs${NC}"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run.sh [options]"
            echo "Options:"
            echo "  --cpu, -c         Force CPU mode (no GPU)"
            echo "  --download, -d    Download model to cache and exit"
            echo "  --debug, -v       Enable verbose debug output"
            echo "  --quiet, -q       Show only warnings and errors"
            echo "  --clean, -cl      Clean minimal UI with no debug info"
            echo "  --terminal-only, -t   Terminal-only logging (no Streamlit UI logging)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            # Pass other arguments to streamlit
            break
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
poetry install

# Check if we need to install specific packages that might cause issues
echo -e "${GREEN}Ensuring all critical dependencies are installed...${NC}"
poetry run pip install --upgrade torch torchvision accelerate

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
export SHOW_DEBUG_UI=$SHOW_DEBUG_UI

# Set terminal-only logging if requested
if [ "$TERMINAL_ONLY" = "1" ]; then
    export TERMINAL_ONLY=1
fi

# Set clean mode if requested
if [ "$CLEAN_OUTPUT" = true ]; then
    echo -e "${YELLOW}Using clean UI with minimal debug information${NC}"
    export DEBUG_LEVEL=0  # ERROR only
    export SHOW_DEBUG_UI=0
fi

# Run the app
echo -e "${GREEN}Starting Streamlit app...${NC}"
echo -e "${YELLOW}If you encounter any issues, try running:${NC}"
echo -e "${YELLOW}./run.sh --cpu${NC}"
echo "------------------------"

# Run with no file watcher to avoid errors
poetry run streamlit run src/ocr_app/app.py --server.fileWatcherType none "$@"