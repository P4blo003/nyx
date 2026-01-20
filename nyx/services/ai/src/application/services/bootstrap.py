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
from application.services.inference import InferenceManager
from infrastructure.config.task_model import TaskModelMapping
from infrastructure.config.triton_server import TritonInferenceServerMapping


# ==============================
# FUNCTIONS
# ==============================

async def bootstrap_ai_service():
    """
    Load initial configuration and initialize AI service singletons.
    """

    # Load task mapping.
    task_mapping_data:Dict[str, Any] = yaml.load_file(path=os.environ.get("TASK_MAPPING", "./config/task_model_mapping.yaml"))
    task_mapping:TaskModelMapping = TaskModelMapping(**task_mapping_data)

    # Load triton servers mapping.
    server_mapping_data:Dict[str, Any] = yaml.load_file(path=os.environ.get("SERVER_MAPPING", "./config/triton_server_mapping.yaml"))
    server_mapping:TritonInferenceServerMapping = TritonInferenceServerMapping(**server_mapping_data)

    # Initialize ModelManager with loaded mappings.
    inference_manager:InferenceManager = InferenceManager.initialize(task_mapping=task_mapping, server_mapping=server_mapping)
    await inference_manager.startup()