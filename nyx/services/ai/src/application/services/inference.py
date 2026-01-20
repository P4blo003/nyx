# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict, Optional, Tuple

# External:
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from shared.singleton import Singleton
from infrastructure.config.task_model import TaskModelMapping
from infrastructure.config.triton_server import TritonInferenceServerMapping
from domain.inference.inference_task import InferenceTask


# ==============================
# CLASSES
# ==============================

class InferenceManager(Singleton):
    """
    Central runtime manager responsible for model lifecycle (load / unload), task to
    model routing, triton server selection and inference execution.
    """

    # ---- Default ---- #

    def __init__(
        self,
        task_mapping:TaskModelMapping,
        server_mapping:TritonInferenceServerMapping
    ) -> None:
        """
        Initialize the manager.

        Args:
            task_mapping (TaskModelMapping): Pre-loaded mapping of tasks to models.
            server_mapping (TritonInferenceServerMapping): Pre-loaded mapping
                of server identifiers to Triton servers.
        """

        # Initialize the class properties.
        self._task_mapping:TaskModelMapping = task_mapping
        self._server_mapping:TritonInferenceServerMapping = server_mapping

        self._clients:Dict[str, InferenceServerClient] = {}


    # ---- Methods ---- #

    def check_availability(
        self,
        task_type:InferenceTask
    ) -> Tuple[str, InferenceServerClient]:
        """
        Validate that a given inference task is supported an return its associated
        model name and Triton client instance.

        Args:
            task_type (InferenceTask): The inference task to validate.

        Raises:
            ValueError: If the task is not supported, not configured, or the server
                client is not initialized.

        Returns:
            Tuple[str, InferenceServerClient]: The model name and initialized Triton
                Inference Server.
        """

        # Checks if the service supports the task.
        if task_type not in self._task_mapping.tasks.keys(): raise ValueError(f"The service doesn't support '{task_type}' task. Only {self._task_mapping.tasks.keys()} are supported")

        # Gets the available models.
        task = self._task_mapping.tasks.get(task_type, None)
        # Checks if there isn't available models for the task.
        if task is None: raise ValueError(f"The service support '{task_type}' task, but there it isn't configured")

        # Gets the inference client for the given server.
        client:Optional[InferenceServerClient] = self._clients.get(task.server, None)
        # Checks if there isn't available clients for the task-model.
        if client is None: raise ValueError(f"The service supports '{task_type}' with model '{task.model}', "
                                            f"but the Triton client for server '{task.server}' is not initialized.")

        return (task.model, client)

    async def startup(self):
        """
        Initialize all Triton Inference Server clients asynchronously.
        """

        # Iterate over all server mappings to initialize server clients.
        for key, data in self._server_mapping.servers.items(): 
            self._clients[key] = InferenceServerClient(url=f"{data.host}:{data.grpc_port}", verbose=False)

    async def load_model_async(
        self,
        task_type:InferenceTask
    ) -> None:
        """
        Load the model corresponding to a given task on its assigned Triton server.

        Args:
            task_type (InferenceTask): The task whose model should be loaded.

        Raises:
            ValueError: If the task is not supported or the server client is unavailable.
        """

        # Checks availability.
        model, client = self.check_availability(task_type=task_type)

        # Use InferenceServerClient initializes for server model to load the model.
        await client.load_model(model_name=model)

    async def unload_model_async(
        self,
        task_type:InferenceTask
    ) -> None:
        """
        Unload the model corresponding to a given task on its assigned Triton server.

        Args:
            task_type (InferenceTask): The task whose model should be unloaded.

        Raises:
            ValueError: If the task is not supported or the server client is unavailable.
        """

        # Checks availability.
        model, client = self.check_availability(task_type=task_type)

        # Use InferenceServerClient initializes for server model to load the model.
        await client.unload_model(model_name=model)