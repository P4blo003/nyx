# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from pydantic import BaseModel
from pydantic import Field

# Internal:


# ==============================
# CLASSES
# ==============================

class WorkersConfig(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    max_workers:int = Field(..., description="")