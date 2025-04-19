#!/usr/bin/env python

import os
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

MODEL_CACHE_DIR = Path("model_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")


def download_model():
    print(f"Downloading model to cache directory: {MODEL_CACHE_DIR.absolute()}")

    hf_cache_dir = MODEL_CACHE_DIR / "models"
    hf_cache_dir.mkdir(exist_ok=True)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ucaslcl/GOT-OCR2_0",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
        force_download=True,
    )

    print("Downloading model (this may take a while)...")
    model = AutoModel.from_pretrained(
        "ucaslcl/GOT-OCR2_0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
        cache_dir=hf_cache_dir,
        force_download=True,
    )

    print("Model and tokenizer successfully downloaded to cache!")
    print(f"Cache location: {MODEL_CACHE_DIR.absolute()}")

    del model
    del tokenizer


if __name__ == "__main__":
    download_model()
