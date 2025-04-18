"""OCR 2.0 Demo App package."""

from ocr_app.download_model import download_model
from ocr_app.image_processor import get_available_images, process_image
from ocr_app.logger import (
    LogLevel,
    debug,
    error,
    info,
    init_logger,
    set_log_level,
    spinner,
    success,
    warning,
)
from ocr_app.model_loader import MODEL_CACHE_DIR, load_model, setup_cache_dir

__all__ = [
    # Download model functionality
    "download_model",
    # Model loading functionality
    "load_model",
    "setup_cache_dir",
    "MODEL_CACHE_DIR",
    # Image processing functionality
    "process_image",
    "get_available_images",
    # Logging functionality
    "init_logger",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "spinner",
    "LogLevel",
    "set_log_level",
]
