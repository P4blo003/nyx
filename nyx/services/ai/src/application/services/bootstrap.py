# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from typing import Dict, Any

# Internal:
from shared.utils import yaml
from infrastructure.triton.config.triton_server import TritonInferenceServerMapping
from infrastructure.triton.triton_context import TritonContext
from infrastructure.triton.triton_registry import TritonRegistry


# ==============================
# FUNCTIONS
# ==============================

async def bootstrap_ai_service():
    """
    Load initial configuration and initialize AI service singletons.
    """

    # Load triton servers mapping.
    server_mapping_data:Dict[str, Any] = yaml.load_file(path=os.environ.get("SERVER_MAPPING", "./config/triton_server_mapping.yaml"))
    server_mapping:TritonInferenceServerMapping = TritonInferenceServerMapping(**server_mapping_data)

    # Initializes singletons.
    TritonContext.initialize(server_mapping=server_mapping)
    TritonRegistry.initialize()
    
    await TritonContext.get().startup()