# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
import sys
# External:
import uvicorn
from fastapi import FastAPI
# Internal:
from utilities import banner
from config import loader
from api.main import create_application


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except to manage errors.
    try:

        # Prints startup information.
        banner.print_startup_message()

        # Loads environment variables.
        loader.load_env()

        # Creates the FastAPI application.
        app:FastAPI = create_application()

        # Run uvicorn.
        uvicorn.run(
            app=app,
            host=os.environ.get("UVICORN_HOST", "localhost"),
            port=int(os.environ.get("UVICORN_PORT", "80"))
        )

        # Ends process with status code.
        sys.exit(0)

    # If an unexpected error occurs.
    except Exception as ex:

        # Prints information.
        print(f"Critical error during main execution: {ex}")

        # Ends process with status code.
        sys.exit(1000)