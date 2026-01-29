# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Internal:
from infrastructure.vector_store.client.interfaces import IAsyncClient


# ==============================
# CLASSES
# ==============================

class DocumentService:
    """
    
    """

    # ---- Methods ---- #

    def __init__(
        self,
        vs_client:IAsyncClient
    ) -> None:
        """
        Initializes the service.

        Args:
            vs_client (IAsyncClient)
        """

        # Initializes the class properties.
        self._vs_client:IAsyncClient = vs_client


    # ---- Methods ---- #

    async def add_document(self) -> None:
        """
        
        """

        pass

    async def delete_document(self) -> None:
        """
        
        """

        pass

    async def retrieve_relevant_documents(self) -> None:
        """
        
        """
        pass