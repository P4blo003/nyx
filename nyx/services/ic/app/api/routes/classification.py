# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
import sys
from typing import List
# External:
from fastapi import APIRouter
# Internal:
from dto.request import ClassificationRequest
from dto.response import ClassificationResponse


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initilize the router.
    router:APIRouter = APIRouter(
        prefix="/s",
        tags=["WebSocket"]
    )

# If an unexpected error ocurred.
except Exception as ex:
    # Prints information.
    print(f"Fatal error at {__name__}: {ex}")
    # End the program.
    sys.exit(1000)


# ==============================
# ENDPOINTS
# ==============================

@router.post(
    "/classification",
    response_model=ClassificationResponse
)
async def chat(
    request:ClassificationRequest
):
    """

    """
    # Try-Except to manage errors.
    try:
        # Gets classification.
        labels:List[str] = []

        return ClassificationResponse(labels=labels)

    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Error handling request: {ex}")