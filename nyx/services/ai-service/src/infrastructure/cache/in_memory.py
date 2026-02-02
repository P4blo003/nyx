# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from asyncio import Lock
from typing import Optional, Dict
from typing import TypeVar

# Internal:
from infrastructure.cache.base import ICache


# ==============================
# TYPES
# ==============================

V = TypeVar("V")


# ==============================
# CLASSES
# ==============================

class InMemoryCache(ICache[V]):
    """
    Stores key-value paris in memory and is suitable for single-process applications.
    It implements standard dictionary-like operations and is fully generic over the
    type of values stored (V).
    """

    # ---- Default ---- #

    def __init__(self) -> None:
        """
        Initializes the in-memory cache.
        """

        # Initializes the class properties.
        self._store:Dict[str, V] = {}
        self._lock:Lock = Lock()

    
    # ---- Methods ---- #

    async def get(self, key: str) -> Optional[V]:
        """
        Returns the value associated with the given key.

        Args:
            key (str): Cache key.

        Returns:
            Optional[V]: Cached value or None if key does not exist.
        """
        
        async with self._lock: return self._store.get(key, None)

    async def get_all(self) -> Dict[str, V]:
        """
        Returns all cache key-value pairs.

        Returns:
            Dict[str, V]: Complete cache content.
        """
        async with self._lock: return self._store

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
        async with self._lock: self._store[key] = value

    async def update(self, values:Dict[str, V]) -> None:
        """
        Updates the cache using dictionary merge semantics.

        Args:
            values (Dict[str, Any]): Key-value pairs to merge.
        """
        async with self._lock: self._store.update(values)

    async def delete(self, key:str) -> None:
        """
        Removes a key from the cache.

        Args:
            key (str): Cache key.
        """
        async with self._lock: self._store.pop(key, None)

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
        async with self._lock: return key in self._store

    async def clear(self) -> None:
        """
        Removes all entries from the cache.
        """
        async with self._lock: self._store.clear()