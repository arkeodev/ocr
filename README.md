# OCR 2.0 Demo App

A Streamlit application for Optical Character Recognition using the GOT-OCR2.0 model.

## Features

- Plain text OCR
- Formatted OCR with HTML rendering
- Support for multi-column documents
- Support for handwritten text
- Support for charts and math equations
- Support for input/output directories
- Local model caching to avoid repeated downloads
- Poetry package management for better dependency handling

## Setup

### Prerequisites

- Python 3.9+ (except version 3.9.7 which is not compatible with Streamlit)
- Poetry (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ocr-2.0-demo.git
cd ocr-2.0-demo
```

2. Install dependencies with Poetry:
```bash
# Install Poetry if you don't have it
pip install poetry

# Install dependencies
poetry install
```

## Running the App

You can run the app directly using Poetry:

```bash
# Run the app
poetry run streamlit run src/ocr_app/app.py

# Download the model first (recommended for first run)
poetry run download-model
```

Or use the convenience script:

```bash
# Run the app
./run.sh

# Download the model first (recommended for first run)
./run.sh --download
```

## Model Caching

By default, the app will download the model to a local cache directory (`model_cache`) the first time it runs. To pre-download the model without running the app:

```bash
# Download the model to the cache directory
poetry run download-model
```

This avoids downloading the model every time you start the app.

## Usage

1. Upload an image containing text or select from the `input_data` directory
2. Select OCR type:
   - `ocr`: Plain text extraction
   - `format`: Preserve formatting
3. Enable multi-crop for high-resolution images
4. Enable HTML rendering to visualize formatted results

## Directory Structure

- `input_data`: Place images here to be processed by the app. Supported formats: jpg, jpeg, png, gif
- `export_data`: Output files (OCR text and HTML renders) will be saved here
- `model_cache`: Downloaded models are stored here
- `src/ocr_app`: Main application code

## Development

This project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add package-name
```

For development dependencies:

```bash
poetry add --group dev package-name
```

To format the code:

```bash
poetry run black src
poetry run isort src
```

## Troubleshooting

If you encounter issues with Python versions, make sure you're using Python 3.9 or newer (but not 3.9.7 specifically, which has known issues with Streamlit).

## Model Information

This app uses the [GOT-OCR2.0](https://huggingface.co/ucaslcl/GOT-OCR2_0) model from Hugging Face.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 