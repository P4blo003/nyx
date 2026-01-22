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
from typing import List, Any, Optional

# External:
import numpy as np

# Internal:
from infrastructure.triton.triton_context import TritonContext
from domain.models.triton_model import TritonModel
from infrastructure.triton.triton_repository import TritonSDK
from interfaces.api.schemas.inference_response import InferenceResult, InferenceResponse


# ==============================
# INTERFACES
# ==============================

class IInferenceService(ABC):
    """
    """

    # ---- Methods ---- #

    @abstractmethod
    async def make_infer(
        self,
        texts:List[str]
    ) -> Any:
        """
        """
        pass


# ==============================
# CLASSES
# ==============================

class InferenceService(IInferenceService):
    """
    """

    # ---- Default ---- #

    def __init__(
        self,
        context:TritonContext,
        triton_sdk:TritonSDK
    ) -> None:
        """
        Initializes the service.
        """

        # Initializes the class properties.
        self._context:TritonContext = context
        self._triton_sdk:TritonSDK = triton_sdk


    # ---- Methods ---- #

    async def make_infer(
        self,
        texts:List[str]
    ) -> Any:
        """
        """
        
        model_name:str = "bge_m3_ensemble"

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
                    
                    # Variable to keep results.
                    results:List[InferenceResult] = []
                    # Iterate over all calculated embeddings.
                    for index, embedding in enumerate(embeddings):
                        results.append(InferenceResult(
                            id=str(index),
                            embedding=embedding.tolist()
                        ))

                    return InferenceResponse(results=results)