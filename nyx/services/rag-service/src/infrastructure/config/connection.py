# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional

# External:
from pydantic import BaseModel
from pydantic import Field


# ==============================
# CLASSES
# ==============================

class ConnectionConfig(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    host:str = Field(..., description="")
    http_port:Optional[int] = Field(None, description="")
    grpc_port:Optional[int] = Field(None, description="")