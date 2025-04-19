import os
import warnings
from pathlib import Path

# Suppress specific warnings about invalid escape sequences in the model code
warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message="invalid escape sequence"
)
warnings.filterwarnings(
    "ignore", message="Sliding Window Attention is enabled but not implemented for"
)

# Set environment variable to handle torch._classes issue
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use minimal GPU memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
os.environ["TERMINAL_ONLY"] = "1"  # Force terminal-only mode

# Try to import the required packages with error handling
try:
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(
        f"Error importing required packages: {str(e)}. Make sure all dependencies are installed with 'poetry install'."
    )

from ocr_app.image_processor import get_available_images, process_image
from ocr_app.logger import debug, error, info, init_logger, spinner, success, warning

# Import from our modules
from ocr_app.model_loader import MODEL_CACHE_DIR, load_model, setup_cache_dir

# Define input/output directories
INPUT_DIR = Path("input_data")
OUTPUT_DIR = Path("export_data")

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup model cache directory
setup_cache_dir(MODEL_CACHE_DIR)


def save_results(text: str, html: str, image_path: Path) -> None:
    """Save OCR results to output directory.

    Args:
        text: Extracted text content
        html: HTML rendered content
        image_path: Original image path (for naming)
    """
    file_stem = image_path.stem

    # Save text result
    output_text = OUTPUT_DIR / f"{file_stem}_ocr.txt"
    with open(output_text, "w") as f:
        f.write(text)

    # Save HTML if available
    if html:
        output_html = OUTPUT_DIR / f"{file_stem}_render.html"
        with open(output_html, "w") as f:
            complete_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OCR Result - {file_stem}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .ocr-result {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h1>OCR Result - {file_stem}</h1>
                <div class="ocr-result">
                    <h2>OCR Text</h2>
                    {html}
                </div>
            </body>
            </html>
            """
            f.write(complete_html)

        debug(f"HTML content length: {len(html)} characters")
        success(f"HTML output saved to {output_html}")

    debug(f"OCR result length: {len(text)} characters")
    success(f"OCR result saved to {output_text}")


def process_all_images(multi_crop: bool = True, render_html: bool = True):
    """Process all images in the input directory.

    Args:
        multi_crop: Whether to use multi-crop for better results
        render_html: Whether to generate HTML output
    """
    # Get input files
    input_files = get_available_images(INPUT_DIR)

    if not input_files:
        info(
            f"No image files found in {INPUT_DIR}. Please add images to this directory."
        )
        return

    # Load the model once to reuse
    with spinner("Loading model... This may take a minute the first time..."):
        model, tokenizer = load_model(MODEL_CACHE_DIR)

    success(f"Found {len(input_files)} image(s) to process")

    # Process each image
    for i, image_path in enumerate(input_files):
        info(f"Processing image {i+1}/{len(input_files)}: {image_path.name}")

        # Process with OCR
        with spinner(f"Processing {image_path.name} with OCR..."):
            ocr_text, html_content = process_image(
                model,
                tokenizer,
                image_path,
                multi_crop=multi_crop,
                render_html=render_html,
            )
            # Save the results
            save_results(ocr_text, html_content, image_path)

    success(f"All {len(input_files)} image(s) processed successfully!")
    success(f"Results saved to {OUTPUT_DIR}")


def main():
    """Main application entry point."""
    # Initialize the logger
    init_logger()

    debug("Application started in terminal-only mode")

    try:
        # Default settings
        multi_crop = True  # Use multi-crop for better results
        render_html = True  # Generate HTML output

        # Process all images with default settings
        process_all_images(multi_crop=multi_crop, render_html=render_html)

    except Exception as e:
        error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
