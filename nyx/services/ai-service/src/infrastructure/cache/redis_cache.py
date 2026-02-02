# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Optional
from typing import Mapping, Dict

# External:
from pydantic import BaseModel
from redis.asyncio import Redis

# Internal:
from domain.ports.cache import ICache


# ==============================
# CLASSES
# ==============================

class RedisCache(ICache):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str,
        port:int,
        decode_responses:bool = True
    ) -> None:
        """
        Initializes the instance.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port
        self._decode_responses:bool = decode_responses
        
        self._r:Redis = Redis(
            host=self._host,
            port=self._port,
            decode_responses=self._decode_responses
        )

    # ---- Methods ---- #

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Returns the value associated with the given key.

        Args:
            key (str): Cache key.

        Returns:
            Optional[V]: Cached value or None if key does not exist.
        """

        return await self._r.get(key)
    
    async def set(
        self,
        key:str,
        value:BaseModel,
        ttl:Optional[int] = None
    ) -> None:
        """
        Inserts or overwrite a cache entry.

        Args:
            key (str): Cache key.
            value (V): Value to store.
        """

        await self._r.set(key, value.model_dump_json(), ex=ttl)

    async def update(
        self,
        values: Mapping[str, BaseModel],
        ttl:Optional[int] = None
    ) -> None:
        """
        Updates the cache using dictionary merge semantics.

        Args:
            values (Dict[str, BaseModel]): Key-value pairs to merge.
        """

        for key, value in values.items():
            await self._r.set(key, value.model_dump_json(), ex=ttl)

    async def delete(self, key: str) -> None:
        """
        Removes a key from the cache.

        Args:
            key (str): Cache key.
        """

        await self._r.delete(key)

    async def exists(self, key: str) -> bool:
        """
        Checks whether a key exists in the cache.

        Args:
            key (str): Cache key.

        Returns:
            bool: True if key exists, False otherwise.
        """

        return await self._r.exists(key)
    
    async def close(self) -> None:
        """
        """

        await self._r.close()