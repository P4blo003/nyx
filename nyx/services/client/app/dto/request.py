# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External
from pydantic import BaseModel, Field
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
        rag_sql (bool): True if want sql rag for the query.
        rag_doc (bool): True if want doc rag for the query.
    """
    # ---- Attributes ---- #

    mtype:MessageType = Field(default=MessageType.STREAM)
    content:str
    rag_sql:bool = Field(default=False)
    rag_doc:bool = Field(default=False)