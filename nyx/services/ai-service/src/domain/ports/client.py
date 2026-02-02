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
    Represents a generic interface for a Triton Inference Server client.

    This interface abstracts any underlying communication protocol (HTTP, gRPC)
    and provides a common entry point for operations requiring a Triton Client.
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

    @abstractmethod
    def get_server_name(self) -> str:
        """
        Retrieves server name associated to this client.
        """
        pass

    @abstractmethod
    async def get_model_repository_index(self) -> Dict[str, Any]:
        """
        Retrieves the index of the model repository from the Triton Inference Server.

        This method requires the server to list all available models, including
        their versions and current state.

        Returns:
            response (Dict[str, Any]): Information retrieved from Triton Inference Server.
        """
        pass

    @abstractmethod
    async def get_model_metadata(
        self,
        model_name:str
    ) -> Dict[str, Any]:
        """
        Retrieves the metadata for the model.

        This typically includes information about the model's inputs, outputs and
        supported platform.

        Args:
            model_name (str): Name of the model.

        Returns:
            response (Dict[str, Any]): The metadata information for the model.
        """
        pass

    @abstractmethod
    async def get_model_config(
        self,
        model_name:str
    ) -> Dict[str, Any]:
        """
        Retrieves the configuration for the model.

        Args:
            model_name (str): Name of the model.

        Returns:
            response (Dict[str, Any]): The configuration information for the model.
        """
        pass

    @abstractmethod
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly load a specific model.

        Args:
            model_name (str): Name of the model to load.
        """
        pass

    @abstractmethod
    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly unload a specific model.

        Args:
            model_name (str): Name of the model to unload.
        """
        pass

class IClientManager(ABC, Generic[T]):
    """
    Represents a generic interface for managing clients.

    This interface defines the contract for creating, accessing, and shitting down
    clients, allowing multiple containers to be managed in a consistent way.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def start(self) -> None:
        """
        Starts all clients.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stops all clients.
        """
        pass

    @abstractmethod
    def get_clients(self) -> Dict[str, T]:
        """
        Retrieves all clients managed by this manager.

        Returns:
            response (Dict[str, T]): A dictionary mapping containing keys to
                their respective client.
        """
        pass

    @abstractmethod
    def get_client(
        self,
        key:str
    ) -> Optional[T]:
        """
        Retrieve a specific client by its key.

        Args:
            key (str): The unique identifier.

        Returns:
            response (Optional[T]): The client associated with the key, or `None` it
                the key does not exist.
        """
        pass