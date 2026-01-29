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


# ==============================
# TYPES
# ==============================

T = TypeVar("T")


# ==============================
# INTERFACES
# ==============================

class IAsyncClient(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    async def close(self) -> None:
        """
        Close the client.
        """
        pass

    @abstractmethod
    async def is_server_alive(self) -> bool:
        """
        Checks if the associated server is alive.

        Returns:
            response (bool): `True` if the server is alive, `False` otherwise.
        """
        pass