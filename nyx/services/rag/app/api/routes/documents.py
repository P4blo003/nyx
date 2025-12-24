# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import APIRouter


# ==============================
# VARIABLES
# ==============================

router:APIRouter|None = None


# ==============================
# INITIALIZATION
# ==============================

# Try-Except to manage errors.
try:

    # Initializes router.
    router = APIRouter(prefix="/documents")

# If an unexpected error occurs.
except Exception as ex:

    # Raises the error.
    raise RuntimeError(f"Unable to initialize {__name__} router: {ex}")


# ==============================
# ENDPOINTS
# ==============================