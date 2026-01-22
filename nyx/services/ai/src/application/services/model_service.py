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
from infrastructure.triton.triton_repository import TritonSDK


# ==============================
# INTERFACES
# ==============================

class IModelService(ABC):
    """
    """

    # ---- Methods ---- #

    @abstractmethod
    async def get_models(self) -> List[TritonModel]:
        """
        """
        pass

    @abstractmethod
    async def get_model_data(
        self,
        model_name:str
    ) -> Optional[TritonModel]:
        """
        """
        pass

    @abstractmethod
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        """
        pass

    @abstractmethod
    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        """
        pass


# ==============================
# CLASSES
# ==============================

class ModelService(IModelService):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        context:TritonContext,
        triton_sdk:TritonSDK
    ) -> None:
        """
        Initializes the service properties.
        """

        # Initializes the class properties.
        self._context:TritonContext = context
        self._triton_sdk:TritonSDK = triton_sdk


    # ---- Methods ---- #

    async def get_models(self) -> List[TritonModel]:
        """ 
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
        """

        # Iterate over all available clients.
        for __, client in self._context.Clients.items():
            # Gets available models.
            models:List[TritonModel] = await self._triton_sdk.get_models(client=client)

            for model in models:
                if model.name == model_name:
                    return await self._triton_sdk.unload_model(client=client, model_name=model_name)