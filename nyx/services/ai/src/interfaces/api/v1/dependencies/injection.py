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


# ==============================
# FUNCTIONS
# ==============================

def get_model_service(request:Request) -> ModelService:
    """
    
    """

    # Get application client manager, model cache and returns
    # the model service.
    return ModelService(
        triton_client_manager=request.app.state.client_manager,
        model_cache=request.app.state.model_cache
    )