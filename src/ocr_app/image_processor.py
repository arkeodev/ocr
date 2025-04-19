import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .logger import debug, error, info, spinner, success, warning


def process_image(
    model: Any,
    tokenizer: Any,
    image_path: Union[str, Path],
    multi_crop: bool = False,
    render_html: bool = True,
) -> Tuple[str, str]:
    try:
        image_path = Path(image_path)
        with spinner(f"Loading image {image_path.name}..."):
            pil_img = Image.open(image_path)
            processed_img = preprocess_image(pil_img)
    except UnidentifiedImageError:
        error(f"Could not identify image file: {image_path}")
        return "Error: Could not identify image file.", ""
    except Exception as e:
        error(f"Error loading image: {e}")
        return f"Error loading image: {e}", ""

    with spinner("Processing image with OCR model..."):
        text, html = run_ocr_on_image(
            model,
            tokenizer,
            processed_img,
            multi_crop=multi_crop,
            render_html=render_html,
        )

    info(f"OCR processing successful: extracted {len(text)} characters")

    return text, html


def preprocess_image(img: Image.Image) -> Image.Image:
    debug(f"Preprocessing image: size={img.size}, mode={img.mode}")

    if img.mode != "RGB":
        debug("Converting image to RGB mode")
        img = img.convert("RGB")

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
    debug(f"Processing single image: size={img.size}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        debug(f"Saving image to temporary file: {temp_path}")
        img.save(temp_path, format="JPEG")

    try:
        ocr_type = "format" if render_html else "ocr"
        with torch.no_grad():
            text = model.chat(tokenizer, temp_path, ocr_type=ocr_type)

        debug(f"Extracted {len(text)} characters of text")

        html = ""
        if render_html:
            debug("Using formatted text output as HTML")
            html = text

        return text, html
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            debug(f"Removed temporary file: {temp_path}")


def _process_with_multi_crop(
    model: Any, tokenizer: Any, img: Image.Image, render_html: bool
) -> Tuple[str, str]:
    debug("Using multi-crop approach")

    try:
        debug("Attempting to use model.chat_crop method")
        ocr_type = "format" if render_html else "ocr"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            debug(f"Saving image to temporary file: {temp_path}")
            img.save(temp_path, format="JPEG")

        try:
            with torch.no_grad():
                text = model.chat_crop(tokenizer, temp_path, ocr_type=ocr_type)

            debug(f"Successfully processed with chat_crop: {len(text)} characters")

            html = ""
            if render_html:
                debug("Using chat_crop formatted output as HTML")
                html = text

            return text, html
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                debug(f"Removed temporary file: {temp_path}")

    except (AttributeError, TypeError) as e:
        debug(f"chat_crop not available ({str(e)}), falling back to manual crop")

    width, height = img.size
    debug(f"Original image dimensions: {width}x{height}")

    crops = []

    if width > height * 1.5:
        debug("Wide image detected, creating horizontal crops")
        overlap = width // 5
        first_width = (width // 2) + overlap
        second_start = (width // 2) - overlap

        crops.append((0, 0, first_width, height))
        crops.append((second_start, 0, width, height))
    elif height > width * 1.5:
        debug("Tall image detected, creating vertical crops")
        overlap = height // 5
        first_height = (height // 2) + overlap
        second_start = (height // 2) - overlap

        crops.append((0, 0, width, first_height))
        crops.append((0, second_start, width, height))
    else:
        debug("Standard image dimensions, processing as a single crop")
        crops.append((0, 0, width, height))

    all_texts = []
    ocr_type = "format" if render_html else "ocr"

    for i, crop_dims in enumerate(crops):
        debug(f"Processing crop {i+1}/{len(crops)}: {crop_dims}")
        cropped_img = img.crop(crop_dims)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            debug(f"Saving crop to temporary file: {temp_path}")
            cropped_img.save(temp_path, format="JPEG")

        try:
            with torch.no_grad():
                crop_text = model.chat(tokenizer, temp_path, ocr_type=ocr_type)

            debug(f"Crop {i+1} extracted {len(crop_text)} characters")
            all_texts.append(crop_text)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                debug(f"Removed temporary file: {temp_path}")

    combined_text = "\n\n".join(all_texts)
    debug(f"Combined text from {len(crops)} crops: {len(combined_text)} characters")

    html = ""
    if render_html:
        debug("Using formatted text as HTML")
        html = combined_text

    return combined_text, html


def generate_html_output(text: str) -> str:
    html = text.replace("\n", "<br>")
    return f"<div style='font-family: Arial, sans-serif;'>{html}</div>"


def get_image_data_uri(image_path: Union[str, Path]) -> str:
    try:
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            data = f.read()

        encoded = base64.b64encode(data).decode()
        file_ext = image_path.suffix.lower()[1:]

        return f"data:image/{file_ext};base64,{encoded}"
    except Exception as e:
        error(f"Error creating image data URI: {e}")
        return ""


def get_available_images(data_dir: Union[str, Path]) -> List[Path]:
    data_dir = Path(data_dir)

    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

    image_files = []
    for ext in extensions:
        image_files.extend(list(data_dir.glob(f"*{ext}")))
        image_files.extend(list(data_dir.glob(f"*{ext.upper()}")))

    image_files = sorted(image_files)

    debug(f"Found {len(image_files)} images in {data_dir}")
    return image_files
