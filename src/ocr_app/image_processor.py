"""
Image processing utilities for the OCR app.
"""

import os
import io
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, cast

import numpy as np
from PIL import Image, ImageOps
import torch
from PIL import UnidentifiedImageError
import streamlit as st

from .logger import debug, info, error, warning, success, spinner, is_clean_mode

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
            model, tokenizer, processed_img, multi_crop=multi_crop, render_html=render_html
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
    
    # Generate OCR output
    with torch.no_grad():
        outputs = model(img, return_tensors=True)
        
    # Extract text
    text = tokenizer.decode(outputs[0][0].tolist(), skip_special_tokens=True)
    debug(f"Extracted {len(text)} characters of text")
    
    # Render HTML if requested
    html = ""
    if render_html:
        debug("Rendering HTML output")
        html = generate_html_output(text)
    
    return text, html

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
    
    # Get image dimensions
    width, height = img.size
    debug(f"Original image dimensions: {width}x{height}")
    
    # Define crop regions
    # We'll create overlapping crops to ensure we don't miss text at the boundaries
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
    
    for i, crop_dims in enumerate(crops):
        debug(f"Processing crop {i+1}/{len(crops)}: {crop_dims}")
        cropped_img = img.crop(crop_dims)
        
        # Generate OCR output for this crop
        with torch.no_grad():
            outputs = model(cropped_img, return_tensors=True)
            
        # Extract text
        crop_text = tokenizer.decode(outputs[0][0].tolist(), skip_special_tokens=True)
        debug(f"Crop {i+1} extracted {len(crop_text)} characters")
        all_texts.append(crop_text)
    
    # Combine texts
    combined_text = "\n\n".join(all_texts)
    debug(f"Combined text from {len(crops)} crops: {len(combined_text)} characters")
    
    # Render HTML if requested
    html = ""
    if render_html:
        debug("Rendering HTML output for combined text")
        html = generate_html_output(combined_text)
    
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