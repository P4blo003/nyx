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
# External:
from fastapi import APIRouter
# Internal:
from dto.request import RetrieveRequest
from dto.response import RetrieveResponse


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initilize the router.
    router:APIRouter = APIRouter(
        prefix="/rag"
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
    "/retrieve",
    response_model=RetrieveResponse
)
async def retrieve(
    request:RetrieveRequest
) -> None:
    """
    Retrieve documents from the RAG system based on a query.

    Args:
        request (RetrieveRequest): The retrieval request.

    Returns:
        RetreiveResponse: Response containing a list of retrieved documents,
            each with their relevance score, content, and metadata.
    """
    pass