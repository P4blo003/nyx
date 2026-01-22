# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from shared.singleton import Singleton
from infrastructure.triton.config.triton_server import TritonInferenceServerMapping


# ==============================
# CLASSES
# ==============================

class TritonContext(Singleton):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        server_mapping:TritonInferenceServerMapping
    ) -> None:
        """
        Initializes the context.

        Args:
            server_mapping (TritonInferenceServerMapping):
        """

        # Initializes the class properties.
        self._mapping:TritonInferenceServerMapping = server_mapping
        self._clients:Dict[str, InferenceServerClient] = {}

    
    # ---- Properties ---- #

    @property
    def Clients(self) -> Dict[str, InferenceServerClient]:
        """
        """

        return self._clients


    # ---- Methods ---- #

    async def startup(self) -> None:
        """
        
        """

        for key, server in self._mapping.servers.items():
            self._clients[key] = InferenceServerClient(url=f"{server.host}:{server.grpc_port}", verbose=False)


    async def close(self) -> None:
        """
        
        """

        for __, server in self._clients.items():
            await server.close()