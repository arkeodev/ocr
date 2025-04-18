import os
# Disable Streamlit file watcher to avoid torch._classes inspection errors
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Tuple, Union, cast, List

# Set environment variable to handle torch._classes issue
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use minimal GPU memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback

# Try to import the required packages with error handling
try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(f"Error importing required packages: {str(e)}. Make sure all dependencies are installed with 'poetry install'.")

# Import from our modules
from ocr_app.model_loader import load_model, setup_cache_dir, MODEL_CACHE_DIR
from ocr_app.image_processor import (
    process_image, 
    get_available_images,
    get_image_data_uri
)
from ocr_app.logger import (
    init_logger, debug, info, success, warning, error, spinner, 
    LogLevel, is_clean_mode
)

# Define input/output directories
INPUT_DIR = Path("input_data")
OUTPUT_DIR = Path("export_data")

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup model cache directory
setup_cache_dir(MODEL_CACHE_DIR)

st.set_page_config(
    page_title="OCR 2.0 Demo App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded" if not is_clean_mode() else "collapsed",
)

def display_app_header():
    """Display the application header and basic information."""
    st.title("OCR 2.0 Demo App")

    # Only show detailed information in sidebar if not in clean mode
    if not is_clean_mode():
        # Display sidebar info about directories
        st.sidebar.title("App Information")
        st.sidebar.write(f"📂 Input directory: `{INPUT_DIR.absolute()}`")
        st.sidebar.write(f"📂 Output directory: `{OUTPUT_DIR.absolute()}`")
        st.sidebar.write(f"📂 Model cache: `{MODEL_CACHE_DIR.absolute()}`")
        
        # Display system info
        st.sidebar.title("System Info")
        st.sidebar.write(f"🖥️ Platform: {sys.platform}")
        st.sidebar.write(f"🐍 Python: {sys.version.split()[0]}")

def display_image(image_path: Path) -> None:
    """Display an image in the Streamlit UI.
    
    Args:
        image_path: Path to the image file
    """
    try:
        debug(f"Displaying image: {image_path.name}")
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except Exception as e:
        error(f"Error displaying image: {str(e)}")
        debug(f"Image display error details: {str(e)}")
        raise

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
            # Add image reference to HTML
            img_uri = get_image_data_uri(image_path)
            complete_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>OCR Result - {file_stem}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .image-container {{ margin-bottom: 20px; }}
                    .ocr-result {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h1>OCR Result - {file_stem}</h1>
                <div class="image-container">
                    <h2>Original Image</h2>
                    <img src="{img_uri}" style="max-width: 100%; max-height: 500px;" />
                </div>
                <div class="ocr-result">
                    <h2>OCR Text</h2>
                    {html}
                </div>
            </body>
            </html>
            """
            f.write(complete_html)
        
        debug(f"HTML content length: {len(html)} characters")
        if not is_clean_mode():
            success(f"HTML output saved to {output_html}")
    
    debug(f"OCR result length: {len(text)} characters")
    success(f"OCR result saved to {output_text}")

def main():
    """Main application entry point."""
    # Initialize the logger
    init_logger()
    
    # Display application header
    display_app_header()
    
    debug("Application started")
    
    try:
        with spinner("Loading model... This may take a minute the first time..."):
            model, tokenizer = load_model(MODEL_CACHE_DIR)
            
        # UI Layout
        if is_clean_mode():
            # Minimal UI for clean mode
            st.sidebar.title("Settings")
            multi_crop = st.sidebar.checkbox(
                "Use Multi-Crop OCR", value=True,
                help="Use this for high-resolution images"
            )
            render_html = st.sidebar.checkbox(
                "Render HTML Output", value=True,
                help="Format results as HTML"
            )
        else:
            # Full UI
            st.sidebar.title("Settings")
            multi_crop = st.sidebar.checkbox(
                "Use Multi-Crop OCR", value=True,
                help="Use this for high-resolution images"
            )
            render_html = st.sidebar.checkbox(
                "Render HTML Output", value=True,
                help="Format results as HTML"
            )

            st.sidebar.markdown("---")
            
            # Debug settings
            if st.sidebar.checkbox("Show Debug Options", value=False):
                log_level = st.sidebar.radio(
                    "Log Level",
                    options=[level.name for level in LogLevel],
                    index=1  # Default to WARNING
                )
                
                # Enable/disable clean mode
                clean_mode = st.sidebar.checkbox(
                    "Clean Mode (Minimal UI)", 
                    value=is_clean_mode()
                )
                
                if st.sidebar.button("Apply Debug Settings"):
                    os.environ["DEBUG_LEVEL"] = str(LogLevel[log_level].value)
                    os.environ["SHOW_DEBUG_UI"] = "0" if clean_mode else "1"
                    st.experimental_rerun()
        
        # Get input files
        input_files = get_available_images(INPUT_DIR)
        
        # Two ways to input files
        tab1, tab2 = st.tabs(["Upload File", "Select from Input Directory"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png", "gif"]
            )
            
            if uploaded_file is not None:
                # Save the uploaded file to the input directory
                input_path = INPUT_DIR / uploaded_file.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                success(f"File saved to input directory: {input_path}")
                
                # Display the image
                st.subheader("Uploaded Image")
                display_image(input_path)
                
                # Process with OCR
                with spinner("Processing image with OCR..."):
                    ocr_text, html_content = process_image(
                        model, tokenizer, input_path, 
                        multi_crop=multi_crop, render_html=render_html
                    )
                    # Save the results
                    save_results(ocr_text, html_content, input_path)
                
                # Display results
                st.subheader("OCR Results")
                
                # Text result
                st.text_area("Extracted Text", ocr_text, height=400)
                
                # HTML rendering
                if render_html and html_content:
                    st.subheader("HTML Visualization")
                    st.components.v1.html(html_content, height=600, scrolling=True)
        
        with tab2:
            if not input_files:
                info(f"No image files found in {INPUT_DIR}. Please add images to this directory.")
            else:
                # Create a selectbox for choosing input files
                selected_file: Optional[Path] = st.selectbox(
                    "Select input file", 
                    options=input_files,
                    format_func=lambda x: x.name
                )
                
                if selected_file:
                    # Display the image
                    st.subheader("Selected Image")
                    display_image(selected_file)
                    
                    # Process button
                    if st.button("Process Image"):
                        # Process with OCR
                        with spinner("Processing image with OCR..."):
                            ocr_text, html_content = process_image(
                                model, tokenizer, selected_file,
                                multi_crop=multi_crop, render_html=render_html
                            )
                            # Save the results
                            save_results(ocr_text, html_content, selected_file)
                        
                        # Display results
                        st.subheader("OCR Results")
                        
                        # Text result
                        st.text_area("Extracted Text", ocr_text, height=400)
                        
                        # HTML rendering
                        if render_html and html_content:
                            st.subheader("HTML Visualization")
                            st.components.v1.html(html_content, height=600, scrolling=True)
                            
                        # Show output paths
                        success(f"Results saved to {OUTPUT_DIR}")
    except Exception as e:
        error(f"An error occurred: {str(e)}")
        info("Please check that all required dependencies are installed with: ```poetry install```")
        
        # Provide more detailed troubleshooting help
        st.markdown("""
        ### Troubleshooting:
        
        If you're seeing errors related to missing modules or dependencies:
        
        ```bash
        # Make sure all dependencies are installed:
        poetry install
        
        # If you have issues with specific packages:
        poetry run pip install --upgrade torch torchvision accelerate
        
        # Run the app with:
        poetry run streamlit run src/ocr_app/app.py
        ```
        
        #### Apple Silicon (M1/M2/M3) specific issues:
        
        If you're encountering issues with your Apple Silicon Mac:
        
        ```bash
        # Run with CPU only (recommended for troubleshooting)
        ./run.sh --cpu
        ```
        """)


if __name__ == "__main__":
    main()
