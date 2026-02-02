# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any
from typing import Dict, List

# External:
from pydantic import BaseModel
from pydantic import Field


class ProcessedChunk(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    text:str = Field(..., description="")
    metadata:Dict[str, Any] = Field(..., description="")

class DocumentEmbedding(ProcessedChunk):
    """
    
    """

    # ---- Attributes ---- #

    vector:List[float] = Field(..., description="")