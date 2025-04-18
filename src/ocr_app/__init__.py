"""OCR 2.0 Demo App package."""

from ocr_app.download_model import download_model
from ocr_app.model_loader import load_model, setup_cache_dir, MODEL_CACHE_DIR
from ocr_app.image_processor import process_image, get_available_images
from ocr_app.logger import (
    init_logger, debug, info, success, warning, error, spinner,
    LogLevel, set_log_level
)

__all__ = [
    # Download model functionality
    'download_model',
    
    # Model loading functionality
    'load_model', 
    'setup_cache_dir',
    'MODEL_CACHE_DIR',
    
    # Image processing functionality
    'process_image',
    'get_available_images',
    
    # Logging functionality
    'init_logger',
    'debug',
    'info',
    'success',
    'warning',
    'error',
    'spinner',
    'LogLevel',
    'set_log_level'
]
