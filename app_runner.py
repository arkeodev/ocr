#!/usr/bin/env python
"""
OCR 2.0 Demo App Runner
A Python script to run the OCR app using Typer
"""
import os
import subprocess
import sys

import typer  # type: ignore
from rich.console import Console  # type: ignore

app = typer.Typer(help="OCR 2.0 Demo App Runner")
console = Console()

# Define option defaults at module level
CPU_DEFAULT = True
DEBUG_DEFAULT = False

# Define Typer options at module level to avoid B008 errors
cpu_option = typer.Option(
    CPU_DEFAULT, "--cpu/--gpu", help="Force CPU mode (recommended for stability)"
)
debug_option = typer.Option(DEBUG_DEFAULT, "--debug", help="Enable debug logging")


def check_poetry():
    """Check if Poetry is installed and install if needed."""
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[yellow]Installing Poetry...[/yellow]")
        subprocess.run(["pip", "install", "poetry"])
        return True


def install_dependencies():
    """Install dependencies using Poetry."""
    console.print("[green]Installing dependencies...[/green]")
    subprocess.run(
        ["poetry", "install"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def setup_apple_silicon():
    """Configure environment variables for Apple Silicon."""
    if sys.platform == "darwin" and "arm" in os.uname().machine:
        console.print("[yellow]Apple Silicon detected[/yellow]")
        console.print("[green]Setting MPS environment vars...[/green]")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        return True
    return False


@app.command()
def run(
    cpu: bool = cpu_option,
    debug: bool = debug_option,
):
    """Run the OCR app with Streamlit UI."""
    # Check for Poetry and install dependencies
    check_poetry()
    install_dependencies()

    # Set up environment variables
    if cpu:
        console.print("[yellow]Using CPU mode[/yellow]")
        os.environ["FORCE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Configure for Apple Silicon if needed
    setup_apple_silicon()

    # Set debug level if requested
    if debug:
        os.environ["DEBUG_LEVEL"] = "3"  # DEBUG level

    # Start the Streamlit app
    console.print("[green]Starting UI...[/green]")
    console.print("------------------------")

    # Build the command
    cmd = [
        "poetry",
        "run",
        "streamlit",
        "run",
        "src/ocr_app/app.py",
        "--server.fileWatcherType",
        "none",
    ]

    # Run the command
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
