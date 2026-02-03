# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import sys
import signal
import asyncio
import logging
import threading
from typing import Optional

# Internal:
from application.services.client_service import AsyncClientService
from domain.ports.client import IAsyncClientService
from infrastructure.grpc.server import GrpcServer
from shared.utilities import logging as logging_config


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
    log.info("Starting AI-Service ...")

    client_service:IAsyncClientService = AsyncClientService(clients={})
    grpc_server:GrpcServer = GrpcServer()
    server_task:Optional[asyncio.Task] = None

    try:

        await client_service.startup()

        server_task = asyncio.create_task(grpc_server.start())

        if sys.platform == "win32":
            def wait_forever():
                threading.Event().wait()

            threading.Thread(target=wait_forever, daemon=True).start()

        await stop_event.wait()

    except (KeyboardInterrupt, asyncio.CancelledError): pass

    finally:

        log.info("Shutting down ...")

        await grpc_server.stop()
        
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError: pass

        await client_service.shutdown()
    
    log.info("AI-Service stopped successfully.")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Runt async main.
    asyncio.run(main=main())