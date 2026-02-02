# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import BinaryIO

# Internal:
from domain.entities.document import ProcessedChunk


# ==============================
# INTERFACES
# ==============================

class IDocumentProcessor(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    def process(self, file:BinaryIO, filename:str) -> List[ProcessedChunk]:
        """
        
        """
        pass