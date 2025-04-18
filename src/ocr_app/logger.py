"""
Logging module for the OCR 2.0 demo app.
Provides utilities for controlling debug information display.
"""

import os
import sys
from enum import Enum
from typing import Optional, Dict, Any, Union, ContextManager, cast
from contextlib import nullcontext

import streamlit as st

# Debug levels
class LogLevel(Enum):
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3

# Global debug level, can be set via environment variable DEBUG_LEVEL
# 0=ERROR only, 1=WARNING+ERROR, 2=INFO+WARNING+ERROR, 3=All messages
DEFAULT_LOG_LEVEL = LogLevel.INFO

# Check if debug is enabled via environment variable
debug_env = os.environ.get("DEBUG_LEVEL", "")
try:
    DEBUG_LEVEL = LogLevel(int(debug_env)) if debug_env else DEFAULT_LOG_LEVEL
except (ValueError, TypeError):
    DEBUG_LEVEL = DEFAULT_LOG_LEVEL

# Check if completely clean mode (no UI elements)
CLEAN_MODE = os.environ.get("DEBUG_LEVEL", "") == "0" and os.environ.get("SHOW_DEBUG_UI", "1") == "0"

# Terminal-only mode (no Streamlit UI output)
TERMINAL_ONLY = os.environ.get("TERMINAL_ONLY", "0") == "1"

# Create a container for debug messages in the sidebar
_debug_container = None

def is_clean_mode() -> bool:
    """Check if running in clean mode with minimal UI."""
    return CLEAN_MODE

def is_terminal_only() -> bool:
    """Check if running in terminal-only mode with no UI logging."""
    return TERMINAL_ONLY

def init_logger():
    """Initialize the logger with a sidebar container for debug messages."""
    global _debug_container
    
    # Skip UI setup in terminal-only mode
    if is_terminal_only():
        print("[INFO] Terminal-only logging mode active", file=sys.stderr)
        return
    
    # Check if debug mode should be shown in UI
    show_debug_ui = os.environ.get("SHOW_DEBUG_UI", "0") == "1"
    
    if show_debug_ui and not is_clean_mode():
        # Create a container in the sidebar for debug messages
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔍 Debug Information", expanded=False):
            st.write(f"Current log level: **{DEBUG_LEVEL.name}**")
            _debug_container = st.container()
            
            # Add log level selection
            selected_level = st.selectbox(
                "Set log level:",
                options=[level.name for level in LogLevel],
                index=DEBUG_LEVEL.value
            )
            
            # Update debug level when changed
            if selected_level != DEBUG_LEVEL.name:
                set_log_level(LogLevel[selected_level])

def set_log_level(level: LogLevel):
    """Set the current log level.
    
    Args:
        level: The log level to set
    """
    global DEBUG_LEVEL
    DEBUG_LEVEL = level
    
    # If debug container exists, update it
    if _debug_container and not is_terminal_only():
        with _debug_container:
            st.write(f"Log level set to: **{DEBUG_LEVEL.name}**")

def log(message: str, level: LogLevel = LogLevel.INFO, ui: bool = True, show_spinner: bool = False):
    """Log a message at the specified level.
    
    Args:
        message: The message to log
        level: The log level for this message
        ui: Whether to also show in the UI
        show_spinner: Whether to show as a spinner
    """
    # Only log if level is sufficient
    if level.value <= DEBUG_LEVEL.value:
        # Print to console
        print(f"[{level.name}] {message}", file=sys.stderr)
        
        # Skip UI output in terminal-only mode
        if is_terminal_only():
            return
            
        # Show in UI if requested and not in clean mode (except for errors)
        if ui and (not is_clean_mode() or level == LogLevel.ERROR):
            if level == LogLevel.ERROR:
                st.error(message)
            elif level == LogLevel.WARNING and not is_clean_mode():
                st.warning(message)
            elif level == LogLevel.INFO and not is_clean_mode():
                if show_spinner:
                    st.spinner(message)
                else:
                    st.info(message)
            elif level == LogLevel.DEBUG and _debug_container and not is_clean_mode():
                with _debug_container:
                    st.text(f"[DEBUG] {message}")

def debug(message: str):
    """Log a debug message.
    
    Args:
        message: The debug message
    """
    log(message, level=LogLevel.DEBUG, ui=True, show_spinner=False)

def info(message: str, show_spinner: bool = False):
    """Log an info message.
    
    Args:
        message: The info message
        show_spinner: Whether to show as a spinner
    """
    log(message, level=LogLevel.INFO, ui=True, show_spinner=show_spinner)

def success(message: str):
    """Log a success message.
    
    Args:
        message: The success message
    """
    print(f"[SUCCESS] {message}", file=sys.stderr)
    
    # Skip UI in terminal-only mode
    if is_terminal_only():
        return
        
    # Always show success messages, even in clean mode
    st.success(message)

def warning(message: str):
    """Log a warning message.
    
    Args:
        message: The warning message
    """
    log(message, level=LogLevel.WARNING, ui=True)

def error(message: str):
    """Log an error message.
    
    Args:
        message: The error message
    """
    log(message, level=LogLevel.ERROR, ui=True)

def spinner(message: str) -> ContextManager[Any]:
    """Show a spinner with the given message.
    
    Args:
        message: The spinner message
        
    Returns:
        A spinner context manager or dummy manager if not showing spinners
    """
    # Skip UI spinner in terminal-only mode
    if is_terminal_only():
        print(f"[SPINNER] {message}", file=sys.stderr)
        return nullcontext()
        
    # Show spinner only if level is sufficient and not in clean mode
    if DEBUG_LEVEL.value >= LogLevel.INFO.value and not is_clean_mode():
        return cast(ContextManager[Any], st.spinner(message))
    else:
        # Return a dummy context manager
        class DummyContextManager:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContextManager()

def hidden_spinner() -> ContextManager:
    """Return a completely hidden spinner.
    
    This is useful for operations that take a long time but
    we don't want to show any spinner in the UI.
    
    Returns:
        A dummy context manager
    """
    # Return a dummy context manager
    return nullcontext() 