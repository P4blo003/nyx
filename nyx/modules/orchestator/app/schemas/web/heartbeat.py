# ==========================================================================================
# Author: Pablo González García.
# Created: 09/12/2025
# Last edited: 09/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
from typing import Literal
# External:
from pydantic import BaseModel, Field
# Internal:
from schemas.web.base import MessageType


# ==============================
# CLASSES
# ==============================

class HeartbeatContent(BaseModel):
    """
    Content of the heartbeat message.
    
    Attributes:
        id (str): Unique identifier of the heartbeat cycle.
    """
    # ---- Attributes ---- #

    id:str = Field(alias='id')

class HeartbeatResponse(BaseModel):
    """
    Represents a heartbeat response wrapper.

    Attributes:
        rtype (Literal[MessageType.HEARTBEAT]): Discriminator for validation.
        timestamp (float): Server timestamp when message was generated.
        content (HeartbeatContent): Heartbeat payload.
    """
    # ---- Attributes ---- #

    rtype:Literal[MessageType.HEARTBEAT] = Field(alias='type')
    timestamp:float = Field(alias='timestamp', default=0, ge=0)
    content:HeartbeatContent = Field(alias='content')