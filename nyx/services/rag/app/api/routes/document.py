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
from fastapi import APIRouter, UploadFile
# Internal:
from dto.request import UpdateRequest, RetrieveRequest
from dto.response import UpdateResponse, RetrieveResponse
from core.logging.facade import Log
from api.dependencies import VECTOR_STORE as vector_store


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initialize the router.
    router:APIRouter = APIRouter(
        prefix="/document"
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

@router.post("/")
async def upload(
    file:UploadFile
):
    """
    Endpoint to upload documents into the RAG system. The document is queued for processing
    and indexing.
    """
    pass

@router.get("/{doc_id}")
async def get_document(
    doc_id:str
):
    """
    Retrieve metadata or information about a document.

    Args:
        doc_id (str): Document unique identifier.
    """
    pass

@router.delete("/{doc_id}")
async def delete_document(
    doc_id:str
):
    """
    Delete a document from the RAG system.

    Args:
        doc_id (str): Document unique identifier.
    """