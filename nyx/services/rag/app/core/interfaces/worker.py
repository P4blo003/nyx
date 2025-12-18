# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod


# ==============================
# INTERFACES
# ==============================

class IBackgroundWorker(ABC):
    """
    
    """
    # ---- Methods ---- #

    @abstractmethod
    async def start(self) -> None:
        """
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        """
        pass