# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Optional
from typing import Dict, List

# External:
from tritonclient.grpc.aio import InferenceServerClient
from tritonclient.grpc.aio import InferInput, InferResult, InferRequestedOutput

# Internal:
from infrastructure.triton.client.base import TritonAsyncClient


# ==============================
# CLASSES
# ==============================

class GrpcAsyncClient(TritonAsyncClient):
    """
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str,
        port:int
    ) -> None:
        """
        Initializes the gRPC async client.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port
        self._cli:InferenceServerClient = InferenceServerClient(url=self.get_server_url(), verbose=False, ssl=False)


    # ---- Methods ---- #

    def get_server_url(self) -> str:
        """
        
        """

        return f"{self._host}:{self._port}"


    async def connect(self) -> None:
        """
        
        """
        
        try:

            response:Any = await self._cli.is_server_live(client_timeout=3)
            if not isinstance(response, bool) or not response: raise ValueError(f"Unexpected response type: {type(response)}. Expected bool.")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton server: {e}") from e
        
    async def disconnect(self) -> None:
        """
        
        """

        try:

            await self._cli.close()

        except Exception as e:
            raise ConnectionError(f"Failed to disconnect from Triton server: {e}") from e
        
    async def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> None:
        """
        
        """

        model_version = model_version.strip() if model_version and model_version and model_version.strip() else None

        try:
            
            config:Optional[Dict[str, Any]] = None

            if model_version is not None:
                config = {
                    "version_policy": {
                        "specific": {
                            "versions": [model_version]
                        }
                    }
                }
            
            if config: await self._cli.load_model(model_name=model_name, config=config)
            else: await self._cli.load_model(model_name=model_name)

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e
        
    async def unload_model(
        self,
        model_name: str,
        unload_dependents: bool = True
    ) -> None:
        """
        
        """

        try:

            await self._cli.unload_model(
                model_name=model_name,
                unload_dependents=unload_dependents
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e
        
    async def make_infer(
        self,
        model_name: str,
        model_version:Optional[str],
        input_data:List[InferInput],
        output_data:List[InferRequestedOutput]
    ) -> Optional[InferResult]:
        """
        """

        try:

            return await self._cli.infer(
                model_name=model_name,
                model_version=model_version if model_version is not None else "",
                inputs=input_data,
                outputs=output_data
            )

        except Exception as e:
            raise RuntimeError(f"Failed to make inference on model '{model_name}': {e}") from e