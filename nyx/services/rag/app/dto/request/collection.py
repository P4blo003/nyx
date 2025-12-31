# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class CreateCollectionRequest(BaseModel):
    """
    Contains request data to create a new collection.

    Args:
        vector_dim (int): Vector dimension.
        distance (str): Distance of similarity.
    """

    # ---- Attributes ---- #

    vector_dim:int
    distance:str