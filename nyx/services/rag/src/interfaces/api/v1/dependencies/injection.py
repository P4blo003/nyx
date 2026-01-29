# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import Request

# Internal:
from application.services.document_service import DocumentService


# ==============================
# FUNCTIONS
# ==============================

def get_document_service(request:Request) -> DocumentService:
    """
    
    """

    return DocumentService(
        vs_client=request.app.state.vs_client
    )