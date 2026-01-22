# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2026
# Last edited: 22/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List, Optional

# External:
import numpy as np

# Internal:
from infrastructure.triton.triton_context import TritonContext
from domain.models.triton_model import TritonModel
from infrastructure.triton.triton_sdk import TritonSDK
from interfaces.api.schemas.inference_response import InferenceResult, InferenceResponse


# ==============================
# INTERFACES
# ==============================

class IInferenceService(ABC):
    """
    Interface that defines the contract for inference services.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def make_infer(
        self,
        texts:List[str]
    ) -> List[InferenceResult]:
        """
        Executes an inference request over a collection of input texts.

        Args:
            texts (List[str]): Lis of input texts to bre processed by the inference model.

        Returns:
            Any: Backend-specific inference result. Concrete implementations should
                return a well-defined response object.
        """
        pass


# ==============================
# CLASSES
# ==============================

class InferenceService(IInferenceService):
    """
    Concrete implementation of `IInferenceService` using Triton Inference Server.
    """

    # ---- Default ---- #

    def __init__(
        self,
        context:TritonContext,
        triton_sdk:TritonSDK
    ) -> None:
        """
        Initializes the service.

        Args:
            context (TritonContext): Runtime context holding initialized Triton Clients.
            triton_sdk (TritonSDK): Responsible for interacting with Triton Server.
        """

        # Initializes the class properties.
        self._context:TritonContext = context
        self._triton_sdk:TritonSDK = triton_sdk


    # ---- Methods ---- #

    async def make_infer(
        self,
        texts:List[str]
    ) -> List[InferenceResult]:
        """
        Performs inference over the provided texts using a Triton-Hosted model.

        The method iterates through all available Triton Clients, discovers deployed
        models, and executes inference once the target model is found.

        Args:
            texts (List[str]): List of the input texts for which embeddings will be generated.

        Returns:
            List[InferenceResult]: List of inference results.
        """
        
        model_name:str = "bge_m3_ensemble"

        # Variable to keep results.
        results:List[InferenceResult] = []

        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            # Gets available models.
            models:List[TritonModel] = await self._triton_sdk.get_models(client=client)

            for model in models:
                if model.name == model_name:
                    # Make the inference.
                    embeddings:Optional[np.ndarray] = await self._triton_sdk.make_infer(
                                        client=client,
                                        model_name=model_name,
                                        texts=texts
                                    )
                    
                    if embeddings is None: embeddings = np.array([])
                    
                    # Iterate over all calculated embeddings.
                    for index, embedding in enumerate(embeddings):
                        results.append(InferenceResult(
                            id=str(index),
                            embedding=embedding.tolist()
                        ))
    
                    return results
        
        raise RuntimeError("Unable to perform inference. No client/model was found for the task.")