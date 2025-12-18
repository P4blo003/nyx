# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 13/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
from pydantic import BaseModel
# Internal
from dto.document import DocumentRetrieved


# ==============================
# CLASSES
# ==============================

class UpdateResponse(BaseModel):
    """"""
    # ---- Attributes ---- #
    pass

class RetrieveResponse(BaseModel):
    """
    Represents the response from a document retrieval operation.

    Attributes:
        results (List[Document]): List of retrieved documents.
    """
    # ---- Attributes ---- #

    results:List[DocumentRetrieved]