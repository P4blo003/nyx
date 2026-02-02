# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import APIRouter


# ==============================
# MAIN
# ==============================

try:
    # Initializes router.
    router:APIRouter = APIRouter(prefix="/inference")

# If an error occurs.
except Exception as ex:

    raise


# ==============================
# ENDPOINTS
# ==============================