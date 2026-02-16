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

# Internal:
from infrastructure.config.base import RunningConfig
from infrastructure.processor.worker_manager import WorkerManager
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

    workerManager:WorkerManager = WorkerManager(running_config=running_config)

    try:

        log.debug("Starting RAG-Service ...")

        await workerManager.startup()

    finally:

        log.debug("Shutting down ...")

        await workerManager.shutdown() 

    log.debug("RAG-Service stopped successfully.")

    
# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Runt async main.
    asyncio.run(main=main())