# ==========================================================================================
# Author: Pablo González García.
# Created: 31/12/2025
# Last edited: 31/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class VectorInfo(BaseModel):
    """
    Contains information about a vector.

    Args:
        name (str): Name of the vector.
        dim (int): Dimension of the vector.
        distance (str): Distance metric used for the vector.
    """

    # ---- Attributes ---- #

    name:str
    dim:int
    distance:str

class CollectionInfo(BaseModel):
    """
    Contains information about a collection.

    Args:
        name (str): Name of the collection.
        vector_info (List[VectorInfo] | VectorInfo): Information about the vectors in the collection.
    """

    # ---- Attributes ---- #

    name:str
    vector_info:List[VectorInfo]