# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional, Dict, List
from typing import Generic, TypeVar

# External:
from fastapi import UploadFile


# ==============================
# INTERFACES
# ==============================

class IDocumentProcessor(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    async def process(
        self,
        file:UploadFile
    ) -> list:
        """
        """
        pass