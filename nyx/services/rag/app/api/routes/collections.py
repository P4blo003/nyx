# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
from fastapi import APIRouter, Request
from fastapi import HTTPException
# Internal:
from transport.connection.vector_store.interface import IVectorStoreConnection
from dto.request.collection import CreateCollectionRequest
from dto.models.collection import CollectionInfo


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
    router = APIRouter(prefix="/collections")

# If an unexpected error occurs.
except Exception as ex:

    # Raises the error.
    raise RuntimeError(f"Unable to initialize {__name__} router: {ex}")


# ==============================
# ENDPOINTS
# ==============================

@router.get("/")
async def get_collections(
    request:Request
) -> List[CollectionInfo]:
    """
    Gets all collections from the vector store.

    Args:
        request (Request): Request context.

    Raises:
        HTTPException: In case the collections couldn't be retrieved.
    """

    # Try-Except to manage errors.
    try:

        # Gets vector store connection.
        vector_store_connection:IVectorStoreConnection = request.app.state.vector_store_connection

        return await vector_store_connection.get_collections()

    # If an unexpected error occurs.
    except Exception as ex:

        # Raises the error.
        raise HTTPException(
            status_code=500
        )


@router.post("/{collection_name}")
async def create_collection(
    request:Request,
    collection_name:str,
    body:CreateCollectionRequest
) -> None:
    """
    Creates a new collection in the vector store.

    Args:
        request (Request): Request context.
        collection_name (str): Name of the collection to create.
        body (CreateCollectionRequest): Dto with request data.

    Raises:
        HTTPException: In case the collection couldn't be created.
    """

    # Try-Except to manage errors.
    try:

        # Gets vector store connection.
        vector_store_connection:IVectorStoreConnection = request.app.state.vector_store_connection

        # Create a new collection in the vector store.
        await vector_store_connection.create_collection(
            collection_name=collection_name,
            vector_dim=body.vector_dim,
            distance=body.distance
        )

    # If an unexpected error occurs.
    except Exception as ex:

        # Raises the error.
        raise HTTPException(
            status_code=500
        )

@router.delete("/{collection_name}")
async def delete_collection(
    request:Request,
    collection_name:str
) -> None:
    """
    Delete the collection from the vector store.

    Args:
        request (Request): Request context.
        collection_name (str): Name of the collection to delete.

    Raises:
        HTTPException: In case the collection couldn't be deleted.
    """

    # Try-Except to manage errors.
    try:

        # Gets vector store connection.
        vector_store_connection:IVectorStoreConnection = request.app.state.vector_store_connection

        # Create a new collection in the vector store.
        await vector_store_connection.delete_collection(
            collection_name=collection_name
        )

    # If an unexpected error occurs.
    except Exception as ex:

        # Raises the error.
        raise HTTPException(
            status_code=500
        )