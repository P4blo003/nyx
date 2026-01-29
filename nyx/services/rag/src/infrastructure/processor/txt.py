# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional, Dict, List
from typing import Generic, TypeVar

# External:
from fastapi import UploadFile
from pydantic import BaseModel, Field

# Internal:
from infrastructure.processor.interfaces import IDocumentProcessor


class InferenceInput(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    id:str = Field(..., description="Id of the input")
    content:Any = Field(..., description="Content of the input")

class InferenceRequest(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    inputs:List[InferenceInput] = Field(..., description="")
    

# ==============================
# CLASSES
# ==============================

class TxtDocumentProcessor(IDocumentProcessor):
    """
    
    """

    # ---- Methods ---- #

    async def process(
        self,
        file: UploadFile
    ) -> list:
        """
        
        """

        # Reads file content.
        content = (await file.read()).decode("utf-8")

        # Divides text in chunks.
        chunks = []
        start = 0
        block_id = 0

        while start < len(content):
            end = min(start + 500, len(content))
            chunk_text = content[start:end]
            chunks.append({
                "id": str(block_id),
                "content": chunk_text
            })
            start += 500 - 200
            block_id += 1

        return chunks