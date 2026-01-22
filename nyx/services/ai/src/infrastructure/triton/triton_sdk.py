# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2025
# Last edited: 22/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List, Optional, Dict, Any

# External:
import numpy as np
from tritonclient.grpc.aio import InferenceServerClient
from tritonclient.grpc.aio import InferInput, InferRequestedOutput

# Internal:
from domain.models.triton_model import TritonModel


# ==============================
# CLASSES
# ==============================

class TritonSDK:
    """
    Adapter responsible for all direct interactions with
    Triton Inference Server.
    """

    # ---- Methods ---- #

    async def get_models(
        self,
        client:InferenceServerClient,
    ) -> List[TritonModel]:
        """
        Retrieves the list of models available in the Triton Inference Server.

        Args:
            client (InferenceServerClient): Asynchronous gRPC client instance.

        Returns:
            List[TritonModel]: List of model exposed by the Triton Server, populated
                with basic identification and state information. Metadata and
                configuration are not included ath this stage.
        """

        # Gets the response from the Triton Inference Server.
        response = await client.get_model_repository_index(as_json=True)
        if response is None: return []

        # Return the processed data.
        return [ TritonModel(
            name=data.get("name", None),
            version=data.get("version", None),
            state=data.get("state", None),
            metadata=None,
            config=None
        ) for data in response.get("models", [])]
    

    async def get_model_metadata(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> Dict[str, Any]:
        """
        Retrieves metadata for specific model.

        Metadata typically includes information about model inputs, outputs,
        datatypes, and shapes.

        Args:
            client (InferenceServerClient): Asynchronous gRPC client instance.
            model_name (str): Name of the model whose metadata is requested.

        Returns:
            Dict[str, Any]: Raw metadata returned by Triton as a JSON-compatible
                dictionary.
        """

        # Gets the response from the Triton Inference Client.
        return await client.get_model_metadata(model_name=model_name, as_json=True)         # type: ignore
    
    async def get_model_config(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> Dict[str, Any]:
        """
        Retrieves the configuration of a specific model.

        The configuration usually includes instance groups, batching configuration,
        optimization settings, and execution parameters.

        Args:
            client (InferenceServerClient): Asynchronous gRPC client instance.
            model_name (str): Name of the model whose configuration is requested.

        Returns:
            Dict[str, Any]: Raw configuration payload returned by Triton as a JSON-compatible
                dictionary.
        """

        # Gets the response from the Triton Inference Client.
        return await client.get_model_config(model_name=model_name, as_json=True)         # type: ignore
    
    async def load_model(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> None:
        """
        Loads a model into memory on the Triton Inference Server.

        Args:
            client (InferenceServerClient): Asynchronous gRPC client instance.
            model_name (str): Name of the model to load.
        """

        # Loads the model in the Triton Inference Server.
        await client.load_model(model_name=model_name)

    async def unload_model(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> None:
        """
        Unloads a model into memory on the Triton Inference Server.

        Args:
            client (InferenceServerClient): Asynchronous gRPC client instance.
            model_name (str): Name of the model to unload.
        """

        # Unloads the mode from the Triton Inference Server.
        await client.unload_model(model_name=model_name)

    async def make_infer(
        self,
        client:InferenceServerClient,
        model_name:str,
        texts:List[str]
    ) -> Optional[np.ndarray]:
        """
        """
        
        # Variable to keep the results.
        embeddings:Optional[np.ndarray] = None

        # Prepare data.
        input_data = np.array(texts, dtype=object).reshape([-1, 1])
        
        # Create InferInput.
        inputs = [InferInput(name="TEXT", shape=input_data.shape, datatype="BYTES")]
        inputs[0].set_data_from_numpy(input_data)

        # Defines outputs.
        outputs = [InferRequestedOutput("EMBEDDING")]

        # Realize the inference.
        inference_response = await client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        if inference_response is not None:
            # Get calculated embeddings from the response.
            embeddings:Optional[np.ndarray] = inference_response.as_numpy("EMBEDDING")

        return embeddings