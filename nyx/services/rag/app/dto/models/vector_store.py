# ==========================================================================================
# Author: Pablo González García.
# Created: 18/12/2025
# Last edited: 18/12/2025
# ==========================================================================================

# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict, Any
from dataclasses import dataclass
# External:
import numpy as np


# ==============================
# CLASSES
# ==============================

class VectorDocument:
    """
    
    Attributes:
        chunk_id:str,
        document_id:str,
        embedding:np.typing.NDArray[np.float32],
        content:str,
        metadata:Dict[str, Any]
    """
    # ---- Attributes ---- #

    chunk_id:str
    document_id:str
    embedding:np.typing.NDArray[np.float32]
    content:str
    metadata:Dict[str, Any]