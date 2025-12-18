# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================

# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict, Any
from dataclasses import dataclass


# ==============================
# CLASSES
# ==============================

@dataclass
class RetrievalResult:
    """
    Model for a single search result.

    Attributes:
        
    """
    # ---- Attributes ---- #

    chunk_id:str
    doc_id:str|None
    content:str|None
    score:float
    metadata:Dict[str, Any]