# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from logging import Logger
from logging import getLogger

# External:
from fastapi import APIRouter, UploadFile, status
from fastapi import Depends
from fastapi import HTTPException

# Internal:
from application.services.document_service import DocumentService
from domain.ports.document import IDocumentProcessor
from infrastructure.processor.unstructured_processor import UnstructuredProcessor
from interfaces.api.v1.dependencies.injection import get_document_service


# ==============================
# MAIN
# ==============================

router:APIRouter = APIRouter(prefix="/documents")
log:Logger = getLogger(__name__)


# ==============================
# ENDPOINTS
# ==============================

@router.post(
    path="/",
    status_code=status.HTTP_200_OK
)
async def add_document(
    file:UploadFile,
    service:DocumentService = Depends(get_document_service)
) -> None:
    """
    
    """

    try:
        
        # Process document.
        chunks = await UnstructuredProcessor().process(file=file, filename="")

        import httpx
        # TODO: Sent documents to calculate embeddings.
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:1080/api/v1/inference/embeddings",
                json={"inputs": chunks}
            )
            response.raise_for_status()
            
            print(response.content)

        # TODO: Awaits to add the new document.
        await service.add_document()

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        log.error(f"Unable to upsert the given document: {ex}")

        # Returns status 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to upsert the given document."
        )