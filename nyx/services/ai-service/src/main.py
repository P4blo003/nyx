# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
import logging

# Internal:
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

    # Prints information.
    log.info("Starting AI-Service ...")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Runt async main.
    asyncio.run(main=main())