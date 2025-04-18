"""
Model loading module for the OCR 2.0 demo app.
This module handles the loading of the OCR model and tokenizer.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st
from ocr_app.logger import debug, info, success, warning, error, spinner

# Set environment variables for Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use minimal GPU memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback

# Default cache directory
MODEL_CACHE_DIR = Path("model_cache")

def is_clean_mode():
    """Check if running in clean mode with minimal UI."""
    return os.environ.get("DEBUG_LEVEL", "") == "0" and os.environ.get("SHOW_DEBUG_UI", "1") == "0"

def setup_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Set up and configure the model cache directory."""
    # Use provided cache dir or default
    cache_dir = cache_dir or MODEL_CACHE_DIR
    
    # Ensure directory exists
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables for model caching
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    
    debug(f"Cache directory set up at: {cache_dir}")
    return cache_dir

@st.cache_resource
def load_model(cache_dir: Optional[Path] = None) -> Tuple:
    """Load the OCR model and tokenizer.
    
    Args:
        cache_dir: Directory to use for model caching. If None, uses default.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Set up cache directory
        cache_dir = setup_cache_dir(cache_dir)
        hf_cache_dir = cache_dir / "models"
        hf_cache_dir.mkdir(exist_ok=True)
        
        # Only show in sidebar if not in clean mode
        clean_mode = is_clean_mode()
        
        # Check for CPU force flag
        force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
        if force_cpu and not clean_mode:
            device_map = "cpu"
            st.sidebar.info("🔧 Forced CPU mode via environment variable")
        # Check for available devices - supporting both CUDA and Apple Silicon (M1/M2/M3)
        elif torch.cuda.is_available():
            device_map = "cuda"
            if not clean_mode:
                st.sidebar.success("🚀 Using CUDA GPU acceleration")
        elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon support
            device_map = "mps"
            if not clean_mode:
                st.sidebar.success("🚀 Using Apple M-series GPU acceleration (MPS)")
        else:
            device_map = "cpu"
            if not clean_mode:
                st.sidebar.warning("⚠️ Running on CPU - processing will be slower")
        
        # Only show version info if not in clean mode    
        if not clean_mode:
            st.sidebar.info(f"PyTorch version: {torch.__version__}")
            st.sidebar.info(f"Device: {device_map}")
        
        # Verify torchvision is installed
        try:
            import torchvision
            debug(f"Using torchvision {torchvision.__version__}")
        except ImportError:
            error("torchvision is not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "torchvision"])
            import torchvision
            info(f"Installed torchvision {torchvision.__version__}")
            
        info(f"Loading model from cache: {cache_dir}")
        
        # Load tokenizer
        with spinner("Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(
                "ucaslcl/GOT-OCR2_0", 
                trust_remote_code=True,
                cache_dir=hf_cache_dir,
                local_files_only=False
            )
            debug("Tokenizer loaded successfully")
        
        # Use CPU-first approach for all device types to avoid any issues
        try:
            with spinner("Loading model to CPU first for stability..."):
                debug("Starting model loading to CPU")
                model = AutoModel.from_pretrained(
                    "ucaslcl/GOT-OCR2_0",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                    use_safetensors=True,
                    pad_token_id=tokenizer.eos_token_id,
                    cache_dir=hf_cache_dir,
                    local_files_only=False
                )
                debug("Model loaded to CPU successfully")
            
            # Then try to move to appropriate device if not forced to CPU
            if device_map != "cpu" and not force_cpu:
                try:
                    with spinner(f"Moving model to {device_map} device..."):
                        model = model.to(torch.device(device_map))
                    success(f"Successfully moved model to {device_map} device")
                except Exception as e:
                    warning(f"Could not use {device_map} device: {str(e)}")
                    info("Staying on CPU for compatibility")
                    debug(f"Device error details: {str(e)}")
            elif force_cpu:
                info("Staying on CPU as requested")
        except Exception as e:
            warning(f"Error with CPU-first approach: {str(e)}")
            info("Trying direct device mapping...")
            debug(f"CPU loading error details: {str(e)}")
            
            # Fallback to direct device mapping
            with spinner(f"Loading model directly to {device_map if not force_cpu else 'cpu'}..."):
                model = AutoModel.from_pretrained(
                    "ucaslcl/GOT-OCR2_0",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=device_map if not force_cpu else "cpu",
                    use_safetensors=True,
                    pad_token_id=tokenizer.eos_token_id,
                    cache_dir=hf_cache_dir,
                    local_files_only=False
                )
        
        model = model.eval()
        success("Model loaded successfully!")
        return model, tokenizer
    
    except Exception as e:
        error(f"Error loading model: {str(e)}")
        
        # Provide more specific help based on error type
        if "torchvision" in str(e):
            info("Please install torchvision: `poetry run pip install torchvision`")
        elif "accelerate" in str(e):
            info("Please install accelerate: `poetry run pip install accelerate`")
        else:
            info("Try installing additional dependencies: `poetry run pip install accelerate torchvision`")
        
        # Apple Silicon specific help
        if "mps" in str(e).lower() or "metal" in str(e).lower() or sys.platform == "darwin":
            warning("""
            For Mac M1/M2/M3 users: Try forcing CPU mode:
            
            ```bash
            # Run with CPU only (recommended for troubleshooting)
            ./run.sh --cpu
            ```
            """)
        
        raise 