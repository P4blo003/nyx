# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
import asyncio

# External:
import grpc.aio


# ==============================
# CLASSES
# ==============================

class GrpcServer:
    """
    
    """

    # ---- Default ---- #

    def __init__(self) -> None:
        """
        Initializes the server.
        """

        # Initializes the class properties.
        self._server:grpc.aio.Server = grpc.aio.server()

        self._log:logging.Logger = logging.getLogger("Server")


    # ---- Methods ---- #

    def setup(
        self,
        host:str = "[::]",
        port:int = 50051
    ) -> None:
        """
        
        """

        self._listen_addr:str = f"{host}:{port}"
        self._server.add_insecure_port(self._listen_addr)

    async def start(self) -> None:
        """
        
        """

        # Awaits for server to start.
        await self._server.start()

        # Log information.
        self._log.info(f"Grpc server running at: {self._listen_addr}.")

    async def start_and_wait(self) -> None:
        """
        
        """

        # Starts server execution and awaits for termination.
        await self.start()
        await self._server.wait_for_termination()
    
    async def stop(
        self,
        grace:int = 10
    ) -> None:
        """
        Stops the gRPC server gracefully.
        """
        self._log.info(f"Stopping gRPC server (grace period: {grace}s)...")

        try:
            await asyncio.shield(self._server.stop(grace=grace))
        finally:
            self._log.info("Grpc server stopped successfully.")