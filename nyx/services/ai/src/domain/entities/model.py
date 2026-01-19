# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2025
# Last edited: 15/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ==============================
# ENUMS
# ==============================

class BackendType(str, Enum):
    """
    Enumeration of supported inference backends.

    Attributes:
        ONNX (str): `ONNX-based` inference backend, where tokenization is managed
            externally by the service.
        VLLM (str): `vLLM-based` inference backend, which typically manages tokenization
            internally as part of the model runtime.
    """

    # ---- Attributes ---- #

    ONNX = "onnx"
    VLLM = "vllm"


# ==============================
# CLASSES
# ==============================

@dataclass(frozen=True)
class DeployedModel:
    """
    Represents a model deployed and managed by the system.
    """

    # ---- Attributes ---- #

    name:str
    backend:BackendType
    version:Optional[str]


    # ---- Methods ---- #

    def requires_tokenizer(self) -> bool:
        """
        Determine whether the model requires an external tokenizer.

        Returns:
            bool: `True` if the model backend requires the service to manage
                and apply a tokenizer, `False` otherwise.
        """
        return self.backend == BackendType.ONNX