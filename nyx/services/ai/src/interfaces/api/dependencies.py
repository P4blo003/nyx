# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2025
# Last edited: 22/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import Depends

# Internal:
from infrastructure.triton.triton_context import TritonContext
from infrastructure.triton.triton_repository import TritonSDK
from application.services.model_service import IModelService, ModelService
from application.services.inference_service import IInferenceService, InferenceService
from infrastructure.triton.triton_registry import TritonRegistry


# ==============================
# FUNCTIONS
# ==============================

def get_model_service(
    context:TritonContext = Depends(TritonContext.get),
    triton_sdk:TritonSDK = Depends()
) -> IModelService:
    """
    """

    return ModelService(
        context=context,
        triton_sdk=triton_sdk
    )

def get_inference_service(
    context:TritonContext = Depends(TritonContext.get),
    triton_sdk:TritonSDK = Depends()
) -> IInferenceService:
    """
    """

    return InferenceService(
        context=context,
        triton_sdk=triton_sdk
    )