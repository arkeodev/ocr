"""
Image processing utilities for the OCR app.
"""

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .logger import debug, error, info, is_clean_mode, spinner, success, warning


def process_image(
    model: Any,
    tokenizer: Any,
    image_path: Union[str, Path],
    multi_crop: bool = False,
    render_html: bool = True,
) -> Tuple[str, str]:
    """Process an image and extract text using the OCR model.

    Args:
        model: The OCR model to use
        tokenizer: The tokenizer for the model
        image_path: Path to the image file
        multi_crop: Whether to use multi-crop approach for better results
        render_html: Whether to render HTML for the output

    Returns:
        Tuple of (extracted text, rendered HTML)
    """
    # Load and preprocess image
    try:
        image_path = Path(image_path)
        with spinner(f"Loading image {image_path.name}..."):
            pil_img = Image.open(image_path)
            # Use a different variable name to avoid type issues
            processed_img = preprocess_image(pil_img)
    except UnidentifiedImageError:
        error(f"Could not identify image file: {image_path}")
        return "Error: Could not identify image file.", ""
    except Exception as e:
        error(f"Error loading image: {e}")
        return f"Error loading image: {e}", ""

    # Process with the model
    with spinner("Processing image with OCR model..."):
        text, html = run_ocr_on_image(
            model,
            tokenizer,
            processed_img,
            multi_crop=multi_crop,
            render_html=render_html,
        )

    # Log success
    message = "OCR processing successful"
    if not is_clean_mode():
        info(f"{message}: extracted {len(text)} characters")

    return text, html


def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess the image for OCR processing.

    Args:
        img: The input image

    Returns:
        Preprocessed image
    """
    debug(f"Preprocessing image: size={img.size}, mode={img.mode}")

    # Convert to RGB if needed
    if img.mode != "RGB":
        debug("Converting image to RGB mode")
        img = img.convert("RGB")

    # Rotate image if needed (based on EXIF)
    debug("Applying EXIF orientation")
    img = ImageOps.exif_transpose(img)

    return img


def run_ocr_on_image(
    model: Any,
    tokenizer: Any,
    img: Image.Image,
    multi_crop: bool = False,
    render_html: bool = True,
) -> Tuple[str, str]:
    """Run OCR on an image using the model.

    Args:
        model: The OCR model to use
        tokenizer: The tokenizer for the model
        img: The preprocessed image
        multi_crop: Whether to use multi-crop approach for better results
        render_html: Whether to render HTML for the output

    Returns:
        Tuple of (extracted text, rendered HTML)
    """
    debug(f"Running OCR with multi_crop={multi_crop}, render_html={render_html}")

    if multi_crop:
        debug("Using multi-crop approach for better results")
        text, html = _process_with_multi_crop(model, tokenizer, img, render_html)
    else:
        debug("Using single image approach")
        text, html = _process_single_image(model, tokenizer, img, render_html)

    return text, html


def _process_single_image(
    model: Any, tokenizer: Any, img: Image.Image, render_html: bool
) -> Tuple[str, str]:
    """Process a single image with the OCR model.

    Args:
        model: The OCR model
        tokenizer: The tokenizer
        img: The image to process
        render_html: Whether to render HTML

    Returns:
        Tuple of (extracted text, rendered HTML)
    """
    debug(f"Processing single image: size={img.size}")

    # GOT-OCR2.0 expects a file path, not an image object
    # Save the image to a temporary file first
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        debug(f"Saving image to temporary file: {temp_path}")
        img.save(temp_path, format="JPEG")

    try:
        # Generate OCR output
        ocr_type = "format" if render_html else "ocr"
        with torch.no_grad():
            # Use the official API as documented with file path
            text = model.chat(tokenizer, temp_path, ocr_type=ocr_type)

        debug(f"Extracted {len(text)} characters of text")

        # If format mode was used and HTML is requested, use the format output directly
        html = ""
        if render_html:
            debug("Using formatted text output as HTML")
            html = text

        return text, html
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            debug(f"Removed temporary file: {temp_path}")


def _process_with_multi_crop(
    model: Any, tokenizer: Any, img: Image.Image, render_html: bool
) -> Tuple[str, str]:
    """Process an image using the multi-crop approach.

    Splits the image into multiple sections for better OCR results.

    Args:
        model: The OCR model
        tokenizer: The tokenizer
        img: The image to process
        render_html: Whether to render HTML

    Returns:
        Tuple of (combined text, rendered HTML)
    """
    debug("Using multi-crop approach")

    # For multi-crop, we should use the dedicated multi-crop method if available
    try:
        debug("Attempting to use model.chat_crop method")
        ocr_type = "format" if render_html else "ocr"

        # GOT-OCR2.0 expects a file path, not an image object
        # Save the image to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            debug(f"Saving image to temporary file: {temp_path}")
            img.save(temp_path, format="JPEG")

        try:
            # Use the official API for multi-crop processing
            with torch.no_grad():
                text = model.chat_crop(tokenizer, temp_path, ocr_type=ocr_type)

            debug(f"Successfully processed with chat_crop: {len(text)} characters")

            # If format mode was used and HTML is requested, use the format output directly
            html = ""
            if render_html:
                debug("Using chat_crop formatted output as HTML")
                html = text

            return text, html
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                debug(f"Removed temporary file: {temp_path}")

    except (AttributeError, TypeError) as e:
        # Fallback to manual crop if chat_crop is not available
        debug(f"chat_crop not available ({str(e)}), falling back to manual crop")

    # Manual crop implementation as fallback
    # Get image dimensions
    width, height = img.size
    debug(f"Original image dimensions: {width}x{height}")

    # Define crop regions
    crops = []

    # Handle different image dimensions
    if width > height * 1.5:
        # Wide image: split horizontally into 2 crops with overlap
        debug("Wide image detected, creating horizontal crops")
        overlap = width // 5  # 20% overlap
        first_width = (width // 2) + overlap
        second_start = (width // 2) - overlap

        crops.append((0, 0, first_width, height))  # Left crop
        crops.append((second_start, 0, width, height))  # Right crop
    elif height > width * 1.5:
        # Tall image: split vertically into 2 crops with overlap
        debug("Tall image detected, creating vertical crops")
        overlap = height // 5  # 20% overlap
        first_height = (height // 2) + overlap
        second_start = (height // 2) - overlap

        crops.append((0, 0, width, first_height))  # Top crop
        crops.append((0, second_start, width, height))  # Bottom crop
    else:
        # Standard image: process whole
        debug("Standard image dimensions, processing as a single crop")
        crops.append((0, 0, width, height))

    # Process each crop
    all_texts = []
    ocr_type = "format" if render_html else "ocr"

    for i, crop_dims in enumerate(crops):
        debug(f"Processing crop {i+1}/{len(crops)}: {crop_dims}")
        cropped_img = img.crop(crop_dims)

        # Save the cropped image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            debug(f"Saving crop to temporary file: {temp_path}")
            cropped_img.save(temp_path, format="JPEG")

        try:
            # Generate OCR output for this crop using the official API
            with torch.no_grad():
                crop_text = model.chat(tokenizer, temp_path, ocr_type=ocr_type)

            debug(f"Crop {i+1} extracted {len(crop_text)} characters")
            all_texts.append(crop_text)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                debug(f"Removed temporary file: {temp_path}")

    # Combine texts
    combined_text = "\n\n".join(all_texts)
    debug(f"Combined text from {len(crops)} crops: {len(combined_text)} characters")

    # For HTML rendering, use the formatted text if in format mode
    html = ""
    if render_html:
        debug("Using formatted text as HTML")
        html = combined_text

    return combined_text, html


def generate_html_output(text: str) -> str:
    """Generate HTML output from OCR text.

    Args:
        text: The OCR text

    Returns:
        HTML formatted string
    """
    # Simple HTML formatting
    html = text.replace("\n", "<br>")
    return f"<div style='font-family: Arial, sans-serif;'>{html}</div>"


def get_image_data_uri(image_path: Union[str, Path]) -> str:
    """Convert an image to a data URI for embedding in HTML.

    Args:
        image_path: Path to the image

    Returns:
        Data URI string
    """
    try:
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            data = f.read()

        encoded = base64.b64encode(data).decode()
        file_ext = image_path.suffix.lower()[1:]  # Remove the dot

        return f"data:image/{file_ext};base64,{encoded}"
    except Exception as e:
        error(f"Error creating image data URI: {e}")
        return ""


def get_available_images(data_dir: Union[str, Path]) -> List[Path]:
    """Get available images in the data directory.

    Args:
        data_dir: Directory to scan for images

    Returns:
        List of image paths
    """
    data_dir = Path(data_dir)

    # Common image extensions
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(data_dir.glob(f"*{ext}")))
        image_files.extend(list(data_dir.glob(f"*{ext.upper()}")))

    # Sort by filename
    image_files = sorted(image_files)

    debug(f"Found {len(image_files)} images in {data_dir}")
    return image_files
