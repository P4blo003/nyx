# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import sys
import logging
# External:
from fastapi import APIRouter
# Internal:
from dto.request import UpdateRequest, RetrieveRequest
from dto.response import UpdateResponse, RetrieveResponse
from core.logging.facade import Log


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initialize the router.
    router:APIRouter = APIRouter(
        prefix="/search"
    )

# If an unexpected error occurred.
except Exception as ex:
    # Prints information.
    logging.critical("Unable to initialize router", exc_info=True)
    # End the program.
    sys.exit(1000)


# ==============================
# ENDPOINTS
# ==============================