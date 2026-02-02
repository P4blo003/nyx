# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional
from typing import Mapping, Dict

# External:
from pydantic import BaseModel


# ==============================
# INTERFACES
# ==============================

class ICache(ABC):
    """
    Represents a generic cache interface for the application.

    This interfaces abstracts the cache implementation (in-memory, Redis, etc.)
    from the services that depend on it.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def get(
        self,
        key:str
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the value associated with the given key.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Dict[str, Any]]: Cached value or None if key does not exist.
        """
        pass

    @abstractmethod
    async def set(
        self,
        key:str,
        value:BaseModel
    ) -> None:
        """
        Inserts or overwrite a cache entry.

        Args:
            key (str): Cache key.
            value (BaseModel): Value to store.
        """
        pass

    @abstractmethod
    async def update(self, values:Mapping[str, BaseModel]) -> None:
        """
        Updates the cache using dictionary merge semantics.

        Args:
            values (Mapping[str, BaseModel]): Key-value pairs to merge.
        """
        pass

    @abstractmethod
    async def delete(self, key:str) -> None:
        """
        Removes a key from the cache.

        Args:
            key (str): Cache key.
        """
        pass

    @abstractmethod
    async def exists(
        self,
        key:str
    ) -> bool:
        """
        Checks whether a key exists in the cache.

        Args:
            key (str): Cache key.

        Returns:
            bool: True if key exists, False otherwise.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        """
        pass