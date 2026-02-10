# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 10/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel
from pydantic import Field


# ==============================
# CLASSES
# ==============================

class TritonTask(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    endpoint:str = Field(..., description="")
    model_name:str = Field(..., description="")