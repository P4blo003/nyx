# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import sys
import signal
import asyncio
import logging
from typing import Optional
from typing import Dict
from pathlib import Path

# Internal:
from application.services.client_service import AsyncClientService
from domain.ports.client import IAsyncClientService, IAsyncClient
from infrastructure.grpc.server import GrpcServer
from infrastructure.triton.config.base import TritonConfig
from infrastructure.triton.config.task import TritonTask
from infrastructure.triton.client.grpc import GrpcAsyncClient
from shared.utilities import logging as logging_config
from shared.utilities import yaml


# ==============================
# FUNCTIONS
# ==============================

async def main() -> None:
    """
    
    """

    # Initializes logging configuration.
    logging_config.setup()

    # Gets core application logger.
    log:logging.Logger = logging.getLogger("app")

    # Wait for termination signals.
    stop_event:asyncio.Event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    if sys.platform != "win32":
        # Linux / macOS: Use signal handlers.
        loop:asyncio.AbstractEventLoop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    # Prints information.
    log.debug("Loading configuration ...")
    
    # Loads configuration.
    triton_config:TritonConfig = TritonConfig(**yaml.load_data(path=Path("./config").joinpath("triton.config.yaml")))

    # Prints information.
    log.debug("Initializing clients and services ...")

    clients:Dict[str, IAsyncClient] = {}
    for name, connection in triton_config.connections.items():
        clients[name] = GrpcAsyncClient(
            host=connection.host,
            port=connection.port,
        )

    tasks:Dict[str, TritonTask] = {}
    for name, task in triton_config.tasks.items():
        if task.connection not in clients:
            log.error(f"Endpoint '{task.connection}' for task '{name}' not found in the configured endpoints. Task '{name}' will be skipped.")
        
        else:
            tasks[name] = task
            log.debug(f"Task '{name}' assigned to endpoint '{task.connection}' with model '{task.model_name}'.")

    client_service:IAsyncClientService = AsyncClientService[IAsyncClient](clients=clients)
    grpc_server:GrpcServer = GrpcServer(client_service=client_service, tasks=tasks)
    server_task:Optional[asyncio.Task] = None

    try:

        log.debug("Starting AI-Service ...")
        
        server_task = asyncio.create_task(grpc_server.start())

        await stop_event.wait()

    except (KeyboardInterrupt, asyncio.CancelledError): pass

    finally:

        log.debug("Shutting down ...")

        await grpc_server.stop()
        
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError: pass
    
    log.debug("AI-Service stopped successfully.")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Runt async main.
    asyncio.run(main=main())