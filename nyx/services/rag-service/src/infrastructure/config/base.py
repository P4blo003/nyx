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
from infrastructure.config.worker import WorkersConfig
from infrastructure.config.connection import ConnectionConfig


# ==============================
# CLASSES
# ==============================

class RunningConfig(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    workers:WorkersConfig = Field(..., description="")
    connections:Dict[str, ConnectionConfig] = Field(..., description="")