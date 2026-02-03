# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 03/02/2026
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

class IAsyncClient(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...