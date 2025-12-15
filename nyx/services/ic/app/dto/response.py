# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class ClassificationResponse(BaseModel):
    """
    
    Attributes:
        List[str]
    """
    # ---- Attributes ---- #
    
    labels:List[str]