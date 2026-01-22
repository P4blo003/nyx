# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 22/01/2026
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
    Context for Triton Inference Server clients.
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
        Returns active Triton gRPC clients.

        The returned dictionary maps a logical server identifier to its
        corresponding `InferenceServerClient` instance.

        Returns:
            Dict[str, InferenceClient]: Active Triton clients indexed
                by server key.
        """

        return self._clients


    # ---- Methods ---- #

    async def startup(self) -> None:
        """
        Initializes Triton Inference Server clients for all configured Triton servers.

        This method should be called during application startup. It creates
        asynchronous fRPC clients using the configuration
        provided in `TritonInferenceServerMapping`.
        """

        for key, server in self._mapping.servers.items():
            self._clients[key] = InferenceServerClient(url=f"{server.host}:{server.grpc_port}", verbose=False)


    async def close(self) -> None:
        """
        Gracefully close all active Triton gRPC clients.

        This method should me invoked during application shutdown to ensure the
        network resources and connections are released properly.
        """

        for __, server in self._clients.items():
            await server.close()