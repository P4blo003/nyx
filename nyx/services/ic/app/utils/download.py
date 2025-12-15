# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from transformers import pipeline


# ==============================
# FUNCTIONS
# ==============================

def download_model(model_uri:str, path:str):
    """
    Try to download a model from hugging face.

    Args:
        model_uri (str): Model identifier.
        path (str): Where the model is downloaded.
    """
    # Try-Except to manage errors.
    try:
        # Downloads the model.
        pipeline(
            "zero-shot-classification",
            model=model_uri,
            cache_dir=path,
            device=0
        )        

    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Unable to download {model_uri}: {ex}")