# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
from typing import Dict

# Internal:
from application.services.worker_service import WorkerService
from infrastructure.ai_service.client.base import AIServiceAsyncClient
from infrastructure.ai_service.grpc.client import AIServiceGrpcAsyncClient
from infrastructure.config.base import RunningConfig
from infrastructure.config.connection import ConnectionConfig
from infrastructure.grpc.server import GrpcServer
from infrastructure.qdrant.client.base import QdrantAsyncClient
from infrastructure.qdrant.client.qdrant import QdrantGrpcAsyncClient
from shared.utilities import logging as logging_config
from shared.utilities import yaml


# ==============================
# FUNCTIONS
# ==============================

async def main():
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
    running_config:RunningConfig = RunningConfig(**yaml.load_data(path=Path("./config").joinpath("running.config.yaml")))

    # Prints information.
    log.debug("Initializing clients and services ...")

    ai_connections_config:Optional[ConnectionConfig] = running_config.connections.get("ai-service", None)
    if ai_connections_config is None:
        raise ValueError("Unable to find ai-service connection configuration")

    if ai_connections_config.grpc_port is None:
        raise ValueError("Unable to find ai-service grpc port configuration")
    
    ai_cli:AIServiceAsyncClient = AIServiceGrpcAsyncClient(
        host=ai_connections_config.host, 
        port=ai_connections_config.grpc_port)

    qdrant_connection_config:Optional[ConnectionConfig] = running_config.connections.get("qdrant", None)
    if qdrant_connection_config is None:
        raise ValueError("Unable to find qdrant connection configuration")
    
    if qdrant_connection_config.http_port is None:
        raise ValueError("Unable to find qdrant http port configuration")
    if qdrant_connection_config.grpc_port is None:
        raise ValueError("Unable to find qdrant grpc port configuration")

    qdrant_cli:QdrantAsyncClient = QdrantGrpcAsyncClient(
        host=qdrant_connection_config.host,
        port=qdrant_connection_config.http_port,
        grpc_port=qdrant_connection_config.grpc_port
    )

    worker_service:WorkerService = WorkerService(running_config=running_config)
    grpc_server:GrpcServer = GrpcServer(worker_service=worker_service, ai_client=ai_cli, qdrant_client=qdrant_cli)
    server_task:Optional[asyncio.Task] = None

    try:

        log.debug("Starting RAG-Service ...")
        
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

    log.debug("RAG-Service stopped successfully.")

    
# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Runt async main.
    asyncio.run(main=main())