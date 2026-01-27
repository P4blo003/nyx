# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Optional, Dict
from typing import Generic, TypeVar


# ==============================
# TYPES
# ==============================

V = TypeVar("V")


# ==============================
# INTERFACES
# ==============================

class ICache(ABC, Generic[V]):
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
    ) -> Optional[V]:
        """
        Returns the value associated with the given key.

        Args:
            key (str): Cache key.

        Returns:
            Optional[V]: Cached value or None if key does not exist.
        """
        pass

    @abstractmethod
    async def get_all(self) -> Dict[str, V]:
        """
        Returns all cache key-value pairs.

        Returns:
            Dict[str, V]: Complete cache content.
        """
        pass

    @abstractmethod
    async def set(
        self,
        key:str,
        value:V
    ) -> None:
        """
        Inserts or overwrite a cache entry.

        Args:
            key (str): Cache key.
            value (V): Value to store.
        """
        pass

    @abstractmethod
    async def update(self, values:Dict[str, V]) -> None:
        """
        Updates the cache using dictionary merge semantics.

        Args:
            values (Dict[str, Any]): Key-value pairs to merge.
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
    async def clear(self) -> None:
        """
        Removes all entries from the cache.
        """
        pass