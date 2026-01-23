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

# External:
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from infrastructure.triton.client import ITritonClient


# ==============================
# CLASSES
# ==============================

class GrpcClient(ITritonClient):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        name:str,
        host:str,
        port:int
    ) -> None:
        """
        Initializes the client.

        Args:
            host (str):
            port (int):
        """

        # Initializes the class properties.
        self._name:str = name
        self._url:str = f"{host}:{port}"
        self._client:InferenceServerClient = InferenceServerClient(
            url=self._url,
            verbose=False
        )


    # ---- Methods ---- #

    def get_name(self) -> str:
        """
        """
        
        return self._name

    async def get_model_repository_index(self) -> Dict[str, Any]:
        """
        Retrieves the index of the model repository from the Triton Inference Server.

        This method queries the server to list all available models, including
        their versions and current state.

        Returns:
            response (Dict[str, Any]): Information retrieved from Triton Inference Server.
        """

        response =  await self._client.get_model_repository_index(as_json=True)

        if response is None: raise RuntimeError(f"Unable to get mode list.")
        if not isinstance(response, dict): raise TypeError(f"Invalid metadata response type: {type(response)}")

        return response
    
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

        response = await self._client.get_model_metadata(model_name=model_name, as_json=True)

        if response is None: raise RuntimeError(f"Unable to get model '{model_name}' metadata.")
        if not isinstance(response, dict): raise TypeError(f"Invalid metadata response type: {type(response)}")

        return response
    
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton server to explicitly load a specific model.

        Args:
            model_name (str): The name of the model to load.
        """

        await self._client.load_model(model_name=model_name)

    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        Requests the Triton Inference Server to unload a specific model.

        Args:
            model_name (str): The name of the model to unload.
        """

        await self._client.unload_model(model_name=model_name)