# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from typing import Any, Dict
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from infrastructure.triton.client.base import AsyncClient


# ==============================
# CLASSES
# ==============================

class GrpcAsyncClient(AsyncClient):
    """
    Asynchronous gRPC client for interacting with a Triton Inference Server.

    This client wraps the Triton Inference Server's gRPC API, providing
    methods to query server status, retrieve model (metadata, configuration),
    and manage model lifecycle (load/unload models) and make inference.
    """

    # ---- Default ---- #

    def __init__(
        self,
        server_name:str,
        host:str,
        port:int
    ) -> None:
        """
        Initializes the client.

        Args:
            server_name (str): Name of the server.
            host (str): Hostname or IP of the server.
            port (int): Port on which the Triton server is listening.
        """

        # Initializes the class properties.
        self._server_name:str = server_name
        self._host:str = host
        self._port:int = port

        self._client:InferenceServerClient = InferenceServerClient(url=f"{self._server_name}:{self._port}")


    # ---- Methods ---- #

    def get_server_name(self) -> str:
        """
        Retrieves server name associated to this client.

        Returns:
            response (str): Name of the associated server.
        """

        return self._server_name

    async def close(self) -> None:
        """
        Close the client.
        """

        await self._client.close()

    async def is_server_alive(self) -> bool:
        """
        Checks if the associated server is alive.

        Returns:
            response (bool): `True` if the server is alive, `False` otherwise.
        """

        is_alive = await self._client.is_server_live()
        return isinstance(is_alive, bool) and is_alive
    
    async def get_model_repository_index(self) -> Dict[str, Any]:
        """
        Retrieves the index of the model repository from the Triton Inference Server.

        This method requires the server to list all available models, including
        their versions and current state.

        Returns:
            response (Dict[str, Any]): Information retrieved from Triton Inference Server.
        """

        response = await self._client.get_model_repository_index() or {}
        # If the model is not a dictionary, raises an error.
        if not isinstance(response, dict): raise ValueError(f"Invalid response type: {type(response)}")
        return response
    
    async def get_model_metadata(
        self,
        model_name: str
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

        response = await self._client.get_model_metadata(model_name=model_name)
        # If the model is not a dictionary, raises an error.
        if not isinstance(response, dict): raise ValueError(f"Invalid response type: {type(response)}")
        return response
    
    async def get_model_config(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Retrieves the configuration for the model.

        Args:
            model_name (str): Name of the model.

        Returns:
            response (Dict[str, Any]): The configuration information for the model.
        """

        response = await self._client.get_model_config(model_name=model_name)
        # If the model is not a dictionary, raises an error.
        if not isinstance(response, dict): raise ValueError(f"Invalid response type: {type(response)}")
        return response
    
    async def load_model(
        self,
        model_name: str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly load a specific model.

        Args:
            model_name (str): Name of the model to load.
        """

        await self._client.load_model(model_name=model_name)

    async def unload_model(
        self,
        model_name: str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly unload a specific model.

        Args:
            model_name (str): Name of the model to unload.
        """

        await self._client.unload_model(model_name=model_name)