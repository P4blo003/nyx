# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class ClassificationRequest(BaseModel):
    """
    
    Attributes:
        query (str): Query to classify.
    """
    # ---- Attributes ---- #

    query:str