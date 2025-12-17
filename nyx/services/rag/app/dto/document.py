# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 17/12/2025
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

class DocumentRetrieved(BaseModel):
    """
    Represents a document retrieved from the RAG system.

    Attributes:
        id (str): Unique identifier for the document.
        score (float): Relevance score of the document. Must be grater
            than or equal to 0.
        text (str): The actual text content of the document.
        metadata (Dict[str,str]): Additional metadata associated with the
            document as key-value pairs.
    """
    # ---- Attributes ---- #

    id:str
    score:float = Field(ge=0)
    text:str
    metadata:Dict[str,str]