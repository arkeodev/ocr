#!/usr/bin/env python
"""
Script to run the OCR 2.0 Demo app with Poetry.
"""
import subprocess
import sys
from pathlib import Path

def run():
    """Run the Streamlit app."""
    # Run streamlit
    cmd = ["streamlit", "run", "src/ocr_app/app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    run() 