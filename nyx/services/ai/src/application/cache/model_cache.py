# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Dict, List

# Internal:
from application.cache.cache_interface import ICache


# ==============================
# INTERFACES
# ==============================

class ModelCache(ICache):
    """
    Represents a generic cache interface for the application.

    This interfaces abstracts the cache implementation (in-memory, Redis, etc.)
    from the services that depend on it.
    """

    # ---- Methods ---- #
    
    def get_all(self) -> Dict[str, Any]:
        """
        Gets all key-values pairs currently stored in the cache.

        Returns:
            Dict[str, Any]: A dictionary where each key is a cache key
                and the corresponding value is the cached object.
        """

        return {}

    def get_keys(self) -> List[str]:
        """
        Gets all keys stored in the cache.

        Returns:
            List[str]: A list of all keys.
        """

        return []

    def get(self, key:str) -> Any:
        """
        Gets the value associated with the specified key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key, or `None` if the key does not exist.
        """

        pass

    def update(self, dict:Dict[str, Any]) -> None:
        """
        """

        pass

    def set(self, key:str, value:Any) -> None:
        """
        Sets or updates the value associated with the specified key.

        Args:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """

        pass

    def delete(self, key:str) -> None:
        """
        Deletes the value associated with the specified key.

        Args:
            key (str): The key to delete.
        """

        pass