# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, List, Dict

# External:
from tritonclient.grpc.aio import InferInput, InferRequestedOutput, InferenceServerClient, InferResult

# Internal:
from infrastructure.triton.client.interfaces import IAsyncClient


# ==============================
# CLASSES
# ==============================

class GrpcAsyncClient(IAsyncClient):
    """
    
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
            server_name (str): Name of the server
            host (str):
            port (int):
        """

        # Initializes the class properties.
        self._server_name:str = server_name
        self._url:str = f"{host}:{port}"
        self._client:InferenceServerClient = InferenceServerClient(
            url=self._url,
            verbose=False
        )


    # ---- Methods ---- #

    async def close(self) -> None:
        """
        Close the client.
        """

        # Awaits to close the client.
        await self._client.close()

    async def is_server_alive(self) -> bool:
        """
        Checks if the associated server is alive.

        Returns:
            response (bool): `True` if the server is alive, `False` otherwise.
        """

        # Awaits for a response from server. 
        is_alive = await self._client.is_server_live()

        # Checks if the model is not bool.
        if not isinstance(is_alive, bool): raise ValueError(f"Invalid response type: {type(is_alive)}")
        
        return is_alive
    
    def get_server_name(self) -> str:
        """
        Retrieves server name associated to this client.
        """
        
        # Returns the server name.
        return self._server_name

    async def get_model_repository_index(self) -> Dict[str, Any]:
        """
        Retrieves the index of the model repository from the Triton Inference Server.

        This method requires the server to list all available models, including
        their versions and current state.

        Returns:
            response (Dict[str, Any]): Information retrieved from Triton Inference Server.
        """

        # Awaits for a response from server.
        response =  await self._client.get_model_repository_index(as_json=True)

        # Checks it there is no response from server.
        if response is None: raise RuntimeError(f"Unable to get mode list.")
        # Checks if the model is not a dictionary.
        if not isinstance(response, dict): raise TypeError(f"Invalid response type: {type(response)}")

        return response
    
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

        # Awaits for a response from server.
        response = await self._client.get_model_metadata(model_name=model_name, as_json=True)

        # Checks it there is no response from server.
        if response is None: raise RuntimeError(f"Unable to get model '{model_name}' metadata.")
        # Checks if the model is not a dictionary.
        if not isinstance(response, dict): raise TypeError(f"Invalid metadata response type: {type(response)}")

        return response
    
    async def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieves the configuration for the model.

        Args:
            model_name (str): Name of the model.

        Returns:
            response (Dict[str, Any]): The configuration information for the model.
        """

        # Awaits for a response from server.
        response = await self._client.get_model_config(model_name=model_name, as_json=True)

        # Checks it there is no response from server.
        if response is None: raise RuntimeError(f"Unable to get model '{model_name}' config.")
        # Checks if the model is not a dictionary.
        if not isinstance(response, dict): raise TypeError(f"Invalid config response type: {type(response)}")

        return response
    
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly load a specific model.

        Args:
            model_name (str): Name of the model to load.
        """

        # Awaits for server to load the model.
        await self._client.load_model(model_name=model_name)

    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        Request the Triton Inference Server to explicitly unload a specific model.

        Args:
            model_name (str): Name of the model to unload.
        """

        # Awaits for server to unload the model.
        await self._client.unload_model(model_name=model_name)

    async def infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        outputs: List[InferRequestedOutput]
    ) -> InferResult:
        """
        
        """

        # Awaits for a response from server.
        response = await self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        # Checks it there is no response from server.
        if response is None: raise RuntimeError(f"Unable to make inference with model '{model_name}' in server '{self._server_name}")
        # Checks if the model is not a dictionary.
        if not isinstance(response, InferResult): raise TypeError(f"Invalid config response type: {type(response)}")

        return response