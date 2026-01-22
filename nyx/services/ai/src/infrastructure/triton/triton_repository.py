# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2025
# Last edited: 22/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List, Optional

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
    ):
        """
        """

        # Gets the response from the Triton Inference Client.
        return await client.get_model_metadata(model_name=model_name, as_json=True)
    
    async def get_model_config(
        self,
        client:InferenceServerClient,
        model_name:str
    ):
        """
        """

        # Gets the response from the Triton Inference Client.
        return await client.get_model_config(model_name=model_name, as_json=True)
    
    async def load_model(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> None:
        """
        """

        # Loads the model in the Triton Inference Server.
        await client.load_model(model_name=model_name)

    async def unload_model(
        self,
        client:InferenceServerClient,
        model_name:str
    ) -> None:
        """
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