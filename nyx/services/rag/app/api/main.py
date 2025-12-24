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
# Internal:
from api.lifecycle import lifespan


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
        app = FastAPI(
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Import routes.
        from api.routes import collections

        # Checks if routes are not initialize properly.
        if collections.router is None: raise RuntimeError("Collection router is not initialized.")

        # Add routes to the application.
        app.include_router(router=collections.router)

        # Return application.
        return app

    # If an unexpected error occurs.
    except Exception as ex:

        # Raises the error.
        raise RuntimeError(f"Unable to initialize FastAPI application: {ex}")