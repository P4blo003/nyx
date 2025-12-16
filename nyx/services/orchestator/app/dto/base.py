# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from enum import Enum


# ==============================
# ENUMS
# ==============================

class MessageType(str, Enum):
    """
    Represents the type of the message.

    Attributes:
        QUERY: Message from the client containing the query.
        STREAM: Message from the server containing a chunk of the response.
    """
    # ---- Attributes ---- #

    QUERY = 'query'
    STREAM = 'stream'