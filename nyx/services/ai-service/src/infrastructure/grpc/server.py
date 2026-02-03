# ==========================================================================================
# Author: Pablo González García.
# Created: 03/02/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging

# External:
import grpc.aio


# ==============================
# CLASSES
# ==============================

class GrpcServer:
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str = "[::]",
        port:int = 50051
    ) -> None:
        """
        Initializes the server.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port
        self._url:str = f"{self._host}:{self._port}"

        self._server:grpc.aio.Server = grpc.aio.server()
        self._server.add_insecure_port(address=self._url)

        self._log:logging.Logger = logging.getLogger("GrpcServer")


    # ---- Methods ---- #

    async def start(self) -> None:
        """
        
        """

        await self._server.start()

        self._log.info(f"Running server at {self._url}")

        await self._server.wait_for_termination()


    async def stop(self, grace:int=5) -> None:
        """

        """

        await self._server.stop(grace=5)

        self._log.info(f"Server stopped")