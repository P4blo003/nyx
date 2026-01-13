# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class TokenizeRequestDto(BaseModel):
    """
    Tokenize request content.

    Args:
        queries (str|List[str]): Single query or array of queries
            to tokenize.
    """

    # ---- Attributes ---- #

    queries:str|List[str]