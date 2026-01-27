# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import Request

# Internal:
from application.services.model_service import ModelService
from application.services.inference_service import InferenceService


# ==============================
# FUNCTIONS
# ==============================

def get_model_service(request:Request) -> ModelService:
    """
    
    """

    # Get application client manager, model cache and returns
    # the model service.
    return ModelService(
        client_manager=request.app.state.client_manager,
        cache_service=request.app.state.cache_service
    )

def get_inference_service(request:Request) -> InferenceService:
    """
    
    """

    # Get application client manager, model cache and returns
    # the inference service.
    return InferenceService(
        client_manager=request.app.state.client_manager,
        cache_service=request.app.state.cache_service
    )