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
# Internal:
from dto.base import MessageType


# ==============================
# CLASSES
# ==============================

class ClientRequest(BaseModel):
    """
    
    Attributes:
        mtype (MessageType): Message's type.
        content (str): Content of the message.
    """
    # ---- Attributes ---- #

    mtype:MessageType
    content:str