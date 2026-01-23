# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional, Dict


# ==============================
# INTERFACES
# ==============================

class ITritonClient(ABC):
    """
    Represents a generic interface for a Triton Inference Server client.

    This interface abstracts any underlying communication protocol (HTTP, gRPC)
    and provides a common entry point for operations requiring a Triton client.
    """

    # ---- Methods ---- #

    @abstractmethod
    def get_name(self) -> str:
        """
        
        """

        pass

    @abstractmethod
    async def get_model_repository_index(self) -> Dict[str, Any]:
        """
        Retrieves the index of the model repository from the Triton Inference Server.

        This method queries the server to list all available models, including
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

        This typically includes information about the model's inputs, outputs
        and supported platform.

        Args:
            model_name (str): Name of the model.
        
        Returns:
            response (Dict[str, Any]): The metadata information for the model.
        """

        pass

    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton server to explicitly load a specific model.

        Args:
            model_name (str): The name of the model to load.
        """

        pass
    
    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        Requests the Triton Inference Server to unload a specific model.

        Args:
            model_name (str): The name of the model to unload.
        """

        pass

class ITritonClientManager(ABC):
    """
    Represents a generic interface for managing Triton Inference Server clients.

    This interface defines the contract for creating, accessing, and shutting down
    clients, allowing multiple containers to be managed in a consistent way.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def startup(self) -> None:
        """
        """

        pass

    @abstractmethod
    def get_clients(self) -> Dict[str, ITritonClient]:
        """
        Retrieve all Triton clients managed by this manager.

        Returns:
            response (Dict[str, ITritonClient]): A dictionary mapping container keys to 
                their respective Triton client.
        """

        pass

    @abstractmethod
    def get_client(
        self,
        key:str
    ) -> Optional[ITritonClient]:
        """
        Retrieve a specific Triton client by its key.

        Args:
            key (str): The unique identifier for the Triton client.

        Returns:
            response (Optional[ITritonClient]): The Triton client instance associated with the key, or
                `None` if the key does not exists.
        """

        pass

    @abstractmethod
    async def shutdown(self):
        """
        Gracefully shutdown all Triton clients managed by this manager.

        This method should close connections and release any resources associated
        with the Triton clients to prevent leaks.
        """

        pass