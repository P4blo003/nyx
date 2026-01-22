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

# Internal:
from infrastructure.triton.triton_context import TritonContext
from domain.models.triton_model import TritonModel
from infrastructure.triton.triton_sdk import TritonSDK


# ==============================
# INTERFACES
# ==============================

class IModelService(ABC):
    """
    Interface tha defines the contract for model management services.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def get_models(self) -> List[TritonModel]:
        """
        Retrieves all models available across configured inference clients.

        Returns:
            List[TritonModel]: Collection of models currently visible to the service.
        """
        pass

    @abstractmethod
    async def get_model_data(
        self,
        model_name:str
    ) -> Optional[TritonModel]:
        """
        Retrieves detailed information for a specific model.

        This includes base model information as well as its metadata and
        configuration as reported by the inference backend.

        Args:
            model_name (str): Nme of the model to retrieve.

        Returns:
            Optional[TritonModel]: Fully populated model instance if found; otherwise `None`.
        """
        pass

    @abstractmethod
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        Load a model into memory on the inference backend.

        Args:
            model_name (str): Name of the model to load.
        """
        pass

    @abstractmethod
    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        Unloads a model from memory on the inference backend.

        Args:
            model_name (str): Name of the model to unload.
        """
        pass


# ==============================
# CLASSES
# ==============================

class ModelService(IModelService):
    """
    Triton-base implementation of the model management service.

    This service coordinates multiple Triton clients, providing a unified view and 
    control plane for model lifecycle operations.
    """

    # ---- Default ---- #

    def __init__(
        self,
        context:TritonContext,
        triton_sdk:TritonSDK
    ) -> None:
        """
        Initializes the service properties.

        Args:
            context (TritonContext): Runtime context holding initialized Triton Clients.
            triton_sdk (TritonSDK): Responsible for interacting with Triton Server.
        """

        # Initializes the class properties.
        self._context:TritonContext = context
        self._triton_sdk:TritonSDK = triton_sdk


    # ---- Methods ---- #

    async def get_models(self) -> List[TritonModel]:
        """
        Retrieve all models available across al Triton clients.

        Returns:
            List[TritonModel]: Aggregated list of models from every configured client.
        """

        result:List[TritonModel] = []
        
        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            result.extend(await self._triton_sdk.get_models(client=client))

        return result
    
    async def get_model_data(
        self,
        model_name:str
    ) -> Optional[TritonModel]:
        """
        Retrieves detailed information for a specific model.

        Args:
            model_name (str): Name of the model retrieve.

        Returns:
            Optional[TritonModel]: Populated model instance if found; otherwise `None`.
        """
        
        result:Optional[TritonModel] = None

        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            # Gets available models.
            models:List[TritonModel] = await self._triton_sdk.get_models(client=client)

            for model in models:
                if model.name == model_name:
                    result = model
                    result.metadata = await self._triton_sdk.get_model_metadata(client=client, model_name=model_name)
                    result.config = await self._triton_sdk.get_model_config(client=client, model_name=model_name)

        return result

    async def load_model(
        self,
        model_name: str
    ) -> None:
        """
        Loads a model into memory on the first Triton Client where it is found.

        Args:
            model_name (str): Nae of the model to load.
        """

        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            # Gets available models.
            models:List[TritonModel] = await self._triton_sdk.get_models(client=client)

            for model in models:
                if model.name == model_name:
                    return await self._triton_sdk.load_model(client=client, model_name=model_name)

    async def unload_model(
        self,
        model_name: str
    ) -> None:
        """
        Unloads a model from memory on the first Triton Client where it is found.

        Args:
            model_name (str): Name of the model to unload.
        """

        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            # Gets available models.
            models:List[TritonModel] = await self._triton_sdk.get_models(client=client)

            for model in models:
                if model.name == model_name:
                    return await self._triton_sdk.unload_model(client=client, model_name=model_name)