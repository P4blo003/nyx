# ==========================================================================================
# Author: Pablo González García.
# Created: 21/12/2025
# Last edited: 21/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from pathlib import Path
from typing import Dict, List
# External:
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer


# ==============================
# CONSTANTS
# ==============================

# Base model repository.
BASE_REPO:Path = Path("./model_repository")
# Models to use.
MODELS:Dict[str, str] = {
    "embedder":"BAAI/bge-large-en-v1.5",
    "llm":"Qwen/Qwen2.5-7B-Instruct",
    "sql":"defog/sqlcoder-7b-2"
}


# ==============================
# FUNCTIONS
# ==============================

def setup_structure():
    """
    Creates triton directory structure.
    """

    # Create models directories.
    for model_name in MODELS.keys():
        (BASE_REPO / f"{model_name}_model" / "1").mkdir(parents=True, exist_ok=True)
    
def download_vllm_models():
    """
    """

    # Iterates over vllm models.
    for model_key in ["llm"]:
        # Prints information.
        print(f"Downloading {MODELS[model_key]} ...")

        # Try-Except to manage errors.
        try:
            # Download the model.
            snapshot_download(
                repo_id=MODELS[model_key],
                local_dir=BASE_REPO / f"{model_key}_model" / "1",
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "pytorch_model.bin"]
            )
        # If an unexpected error occurs.
        except Exception as ex:
            # Prints errors.
            print(f"Unable to download ({MODELS[model_key]}): {ex}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    
    # Try-Except to manage errors.
    try:

        # Setup directory structure.
        setup_structure()
        # Download vllm models.
        download_vllm_models()

    # If an unexpected error occurs.
    except Exception as ex:
        # Prints information.
        print(f"Critical error preparing models: {ex}")