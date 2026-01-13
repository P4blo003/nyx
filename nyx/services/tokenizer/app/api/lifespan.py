# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from core.tokenizer import TokenizerService


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    """

    # Try-Except to manage errors.
    try:

        # Loads environment variables.
        model_dir:str = os.environ.get("TOKENIZER_MODEL_PATH", "NaN")

        # Checks if the values are correct.
        if model_dir == "NaN": raise ValueError("No tokenizer model was given.")

        # Initialize application tokenizer.
        app.state.tokenizer = TokenizerService(model_dir=model_dir)
        
        yield

    # Executes finally.
    finally:

        # Free resources.
        pass