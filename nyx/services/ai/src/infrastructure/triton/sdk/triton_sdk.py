# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Dict, List

# Internal:
from domain.models.model import TritonModel
from infrastructure.triton.client import ITritonClient


# ==============================
# CLASSES
# ==============================

class TritonSdk:
    """
    
    """

    # ---- Methods ---- #

    async def get_models(
        self,
        client:ITritonClient
    ) -> Dict[str, List[TritonModel]]:
        """
        Retrieve all models from a Triton Inference Server and parse them into domain models.

        This method queries the server for the model repository index, then converts each
        entry into a `TritonModel` suitable for application and API layers. Metadata and config
        are not loaded by default, use other methods for detailed information.

        Args:
            client (ITritonClient): A Triton client obtained from the `ITritonClientManager`.

        Returns:
            response (Dict[str, List[TritonModel]]): A list of models available on the server. Empty if no
                models are found or the server returns an empty response.
        """

        # Await to get response from Triton Inference Server.
        response:Dict[str, Any] = await client.get_model_repository_index() or {}

        # If the response is empty.
        if not response: return {}

        # Process the response.
        models:List[TritonModel] =  [TritonModel(
            name=data.get("name", "None"),
            version=data.get("version", "None"),
            metadata=None,
            config=None
        ) for data in response.get("models", [])]

        return {client.get_name():models}
    
    async def load_model(
        self,
        client:ITritonClient,
        model_name:str
    ) -> None:
        """
        
        """

        # Loads model.
        await client.load_model(model_name=model_name)

    async def unload_model(
        self,
        client:ITritonClient,
        model_name:str
    ) -> None:
        """
        
        """

        # Loads model.
        await client.unload_model(model_name=model_name)