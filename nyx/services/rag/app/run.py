# ==========================================================================================
# Author: Pablo González García.
# Created: 23/12/2025
# Last edited: 23/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
import uvicorn
from fastapi import FastAPI
# Internal:
from config import loader as cfg_loader
import api


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    
    # Try-Except to manage errors.
    try:
        
        # Load environment variables.
        cfg_loader.load_env()

        # Initializes FastAPI.
        app:FastAPI = api.setup()

        # Runs uvicorn.
        uvicorn.run(app=app)

    # If an unexpected error occurs.
    except Exception as ex:
        
        # Print error.
        print(f"CRITICAL ERROR: {ex}")