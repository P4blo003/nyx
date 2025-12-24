# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import FastAPI


# ==============================
# FUNCTIONS
# ==============================

def create_application() -> FastAPI:
    """
    Creates a new FastAPI application.
    """

    # Try-Except to manage errors.
    try:

        # Initialize Fast API application.
        app = FastAPI()

        # Import routes.
        from api.routes import documents

        # Checks if routes are not initialize properly.
        if documents.router is None: raise RuntimeError("Documents router is not initialized.")

        # Add routes to the application.
        app.include_router(router=documents.router)

        # Return application.
        return app

    # If an unexpected error occurs.
    except Exception as ex:

        # Raises the error.
        raise RuntimeError(f"Unable to initialize FastAPI application: {ex}")