# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List, Dict


# ==============================
# INTERFACES
# ==============================

class IModelRegistryCache(ABC):
    """
    """

    # ---- Methods ---- #
    
    @abstractmethod
    async def get_models(self):
        """
        """
        pass

    @abstractmethod
    async def get_location(
        self,
        model_name:str
    ):
        """
        """
        pass


# ==============================
# CLASSES
# ==============================

class InMemoryModelRegistryCache(IModelRegistryCache):
    """
    """

    # ---- Default ---- #

    def __init__(
        self
    ) -> None:
        """
        Initializes the cache.
        """

        # Initializes the class properties.
        self._models:List = []
        self._locations:Dict[str, str] = {}