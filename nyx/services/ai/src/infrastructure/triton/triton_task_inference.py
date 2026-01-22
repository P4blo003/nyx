# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2025
# Last edited: 22/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any


# ==============================
# INTERFACES
# ==============================

class IInferenceTask:
    """
    Interface for a generic inference task. Any specific inference
    must implement this interface.
    """

    # ---- Methods ---- #

    async def run(
        self,
        inputs:Any
    ) -> Any:
        """
        """
        pass


# ==============================
# CLASSES
# ==============================

class EmbeddingInference(IInferenceTask):
    """
    """
    
    # ---- Methods ---- #

    async def run(self, inputs: Any) -> Any:
        """"""
        pass