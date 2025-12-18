# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 13/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict
# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class UpdateRequest(BaseModel):
    """"""
    # ---- Attributes ---- #
    pass

class RetrieveRequest(BaseModel):
    """
    Represents a request to retrieve documents from the RAG system.

    Attributes:
        query (str): The search query string used to find relevant documents.
        top_k (int): Number of top results to return. Must be at least 1.
        filters (Dict[str,str]): Optional metadata filters to apply to search.
    """
    # ---- Attributes ---- #

    query:str
    top_k:int = Field(ge=1)
    filters:Dict[str,str]