import os
import warnings
from pathlib import Path

# Try to import the required packages with error handling
try:
    import streamlit as st
    from PIL import Image

    # AutoModel and AutoTokenizer are imported in model_loader
except ImportError as e:
    raise ImportError(
        f"Error importing required packages: {str(e)}. "
        f"Make sure all dependencies are installed with 'poetry install'."
    )

# Import our modules
from ocr_app.image_processor import get_available_images, process_image
from ocr_app.logger import error, init_logger
from ocr_app.model_loader import MODEL_CACHE_DIR, load_model, setup_cache_dir

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

# Define input/output directories
INPUT_DIR = Path("input_data")
OUTPUT_DIR = Path("export_data")

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup model cache directory
setup_cache_dir(MODEL_CACHE_DIR)


def save_results(text: str, html: str, image_path: Path) -> tuple[Path, Path | None]:
    """Save OCR results to output directory.

    Args:
        text: Extracted text content
        html: HTML rendered content
        image_path: Original image path (for naming)

    Returns:
        Tuple of (text_file_path, html_file_path) where html_file_path may be None
    """
    file_stem = image_path.stem

    # Save text result
    output_text = OUTPUT_DIR / f"{file_stem}_ocr.txt"
    with open(output_text, "w") as f:
        f.write(text)

    # Save HTML if available
    output_html = None
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

    return output_text, output_html


@st.cache_resource
def load_cached_model():
    """Load the model with Streamlit caching."""
    with st.spinner("Loading model... This may take a minute the first time..."):
        model, tokenizer = load_model(MODEL_CACHE_DIR)
    st.success("Model loaded successfully!")
    return model, tokenizer


def display_app_header():
    """Display the application header."""
    st.title("OCR 2.0 Demo")
    st.markdown(
        "Optical Character Recognition using GOT-OCR2.0. Upload or select an image."
    )

    # Add a sidebar for settings
    st.sidebar.title("Settings")


def display_image(image_path):
    """Display the input image."""
    try:
        image = Image.open(image_path)
        st.image(
            image, caption=f"Input Image: {image_path.name}", use_container_width=True
        )
        return True
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return False


def process_uploaded_image(model, tokenizer, temp_path, multi_crop, render_html):
    """Process an uploaded image."""
    with st.spinner(f"Processing {temp_path.name}..."):
        ocr_text, html_content = process_image(
            model,
            tokenizer,
            temp_path,
            multi_crop=multi_crop,
            render_html=render_html,
        )

        # Save results
        text_file, html_file = save_results(ocr_text, html_content, temp_path)

        # Display results
        st.success(f"Complete! Saved to {OUTPUT_DIR}")

        # Show extracted text
        st.subheader("Extracted Text")
        st.text_area("OCR Result", ocr_text, height=300)

        # Show HTML if generated
        if html_content and render_html:
            st.subheader("Formatted Output")
            st.markdown(html_content, unsafe_allow_html=True)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                with open(text_file, "r", encoding="utf-8") as text_reader:
                    st.download_button(
                        "Download Text",
                        text_reader.read(),
                        file_name=f"{temp_path.stem}_ocr.txt",
                    )
            with col2:
                if html_file:
                    with open(html_file, "r", encoding="utf-8") as html_reader:
                        st.download_button(
                            "Download HTML",
                            html_reader.read(),
                            file_name=f"{temp_path.stem}_render.html",
                        )


def main():
    """Main Streamlit application."""
    # Initialize the logger
    init_logger()

    # Display app header
    display_app_header()

    try:
        # Load model
        model, tokenizer = load_cached_model()

        # Sidebar settings
        st.sidebar.header("OCR Options")
        multi_crop = st.sidebar.checkbox(
            "Use Multi-Crop (better for complex layouts)", value=True
        )
        render_html = st.sidebar.checkbox("Generate HTML Output", value=True)

        # Get available images
        input_files = get_available_images(INPUT_DIR)

        # Create tabs for upload or selection
        tab1, tab2 = st.tabs(["Upload Image", "Select from Directory"])

        with tab1:
            st.header("Upload an Image")
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png", "gif"]
            )

            if uploaded_file is not None:
                # Save the uploaded file temporarily
                temp_path = INPUT_DIR / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Display the image
                if display_image(temp_path):
                    # Process button
                    if st.button("Process Image", key="process_uploaded"):
                        process_uploaded_image(
                            model, tokenizer, temp_path, multi_crop, render_html
                        )

        with tab2:
            st.header("Select from Input Directory")

            if not input_files:
                st.info(
                    f"No image files found in {INPUT_DIR}. "
                    f"Please add images to this directory or upload one."
                )
            else:
                # Select image from dropdown
                image_options = [f.name for f in input_files]
                selected_image = st.selectbox(
                    "Select an image to process:", image_options
                )

                # Get the full path
                selected_path = INPUT_DIR / selected_image

                # Display the image
                if display_image(selected_path):
                    # Process button
                    if st.button("Process Image", key="process_selected"):
                        process_uploaded_image(
                            model, tokenizer, selected_path, multi_crop, render_html
                        )

        # Add a footer
        st.markdown("---")
        st.markdown("OCR 2.0 Demo - GOT-OCR2.0 model")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        error(f"Error details: {str(e)}")


if __name__ == "__main__":
    main()
