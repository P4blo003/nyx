# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Optional, Dict, List

# External:
import numpy as np
from tritonclient.grpc.aio import InferInput, InferRequestedOutput, InferResult

# Internal:
from domain.entities.triton.models import TritonModel
from domain.enums.model_state import ModelState
from infrastructure.triton.client.interfaces import IAsyncClient


# ==============================
# CLASSES
# ==============================

class SDK:
    """
    
    """

    # ---- Methods ---- #

    @classmethod
    async def get_models(
        cls,
        client:IAsyncClient
    ) -> Dict[str, List[TritonModel]]:
        """
        Retrieve all models from a Triton Inference Server.

        This method queries the server for the model repository index. Metadata and config are
        not loaded by default, use other methods for detailed information.

        Args:
            client (IAsyncTritonClient): A client to make the query.

        Returns:
            response (Dict[str, List[TritonModel]]): A list of models available on the server. Empty if no
                models are found or the server returns an empty response.

        Raises:
            RuntimeError: If it's impossible to load models from the specified client.
        """

        # Awaits to get response from Triton Inference Server.
        response:Dict[str, Any] = await client.get_model_repository_index() or {}

        # If the response is empty, returns an empty dictionary.
        if not response: return {}

        try:
            # Variable to save all retrieved models.
            models:List[TritonModel] = []

            # Iterate over all retrieved models.
            for model in response.get("models", []):

                # Normalize data.
                name:Optional[str] = model.get("name", None)
                version:Optional[str] = model.get("version", None)
                state:Optional[ModelState] = model.get("state", ModelState.UNKNOWN)
                reason:Optional[str] = model.get("reason", None)

                # Checks if one of the is None.
                if name is None: raise ValueError("The 'name' value is None.")

                # Add the model to the list.
                models.append(TritonModel(
                    name=name,
                    version=version,
                    state=state,
                    reason=reason,
                    metadata=None,
                    config=None
                ))

            # Returns the dictionary with pair key-value, where key is the name of
            # the server (assigned to client), and the value is the list of
            # available models.
            return {client.get_server_name():models}

        # If an error occurs.
        except Exception as ex:
            # Raise a new error.
            raise RuntimeError(f"SDK: Unable to get models from {client.get_server_name()}: {ex}")
    
    @classmethod
    async def load_model(
        cls,
        client:IAsyncClient,
        model_name:str
    ) -> None:
        """
        Load a model into a Triton Inference Server.

        This method sends an asynchronous request to the given client to load the specified model.
        
        Args:
            client (IAsyncTritonClient): A client to make the query.
            model_name (str): The name of the model to load.

        Raises:
            RuntimeError: If the model cannot be loaded on the specified server.
        """

        try:
            # Awaits to load the model.
            await client.load_model(model_name=model_name)

        # If an error occurs.
        except Exception as ex:
            # Raise a new error.
            raise RuntimeError(f"SDK: Unable to load model {model_name} in server {client.get_server_name()}: {ex}")
        
    @classmethod
    async def unload_model(
        cls,
        client:IAsyncClient,
        model_name:str
    ) -> None:
        """
        Unload a model from a Triton Inference Server.

        This method sends an asynchronous request to the given client to unload the specified model.
        
        Args:
            client (IAsyncTritonClient): A client to make the query.
            model_name (str): The name of the model to unload.

        Raises:
            RuntimeError: If the model cannot be unloaded on the specified server.
        """

        try:
            # Awaits to load the model.
            await client.unload_model(model_name=model_name)

        # If an error occurs.
        except Exception as ex:
            # Raise a new error.
            raise RuntimeError(f"SDK: Unable to unload model {model_name} from server {client.get_server_name()}: {ex}")
        
    async def make_infer(
        self,
        client:IAsyncClient,
        model_name:str,
        inputs:List
    ) -> Optional[np.ndarray]:
        """
        """
        
        # Prepare data to sent.
        input_data:np.ndarray = np.array(inputs, dtype=object).reshape([-1, 1])

        # Create infer input to sent.
        inputs_array:List[InferInput] = [InferInput(
            name="TEXT",
            shape=input_data.shape,
            datatype="BYTES"
        )]
        inputs_array[0].set_data_from_numpy(input_data)

        # Defines outputs.
        outputs:List[InferRequestedOutput] = [InferRequestedOutput("EMBEDDING")]

        # Realize the inference.
        response:InferResult = await client.infer(
            model_name=model_name,
            inputs=inputs_array,
            outputs=outputs
        )

        return response.as_numpy("EMBEDDING")