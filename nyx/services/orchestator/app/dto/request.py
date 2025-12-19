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
    Data transfer object representing a client request.

    Encapsulates all information needed to process a client's chat message,
    including message type, content, and RAG (Retrieval-Augmented Generation)
    preferences for both SQL and document sources.

    Attributes:
        mtype (MessageType): The type of message being sent.
        content (str): The actual message content from the client.
        rag_sql (bool): Whether to use SQL RAG for the query. Defaults to False.
        rag_doc (bool): Whether to use document RAG for the query. Defaults to False.
    """
    # ---- Attributes ---- #

    mtype:MessageType = Field(default=MessageType.STREAM)
    content:str
    rag_sql:bool = Field(default=False)
    rag_doc:bool = Field(default=False)