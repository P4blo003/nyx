# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio

# External:
import grpc.aio

# Internal:
from interfaces.grpc.server import GrpcServer
from shared.utilities import logging


# ==============================
# FUNCTIONS
# ==============================

async def main():
    """
    
    """

    # Setup logging.
    logging.setup_logging()

    # Creates async gRPC server.
    server:GrpcServer = GrpcServer()
    server.setup()

    try:
        # Starts server and awaits for termination.
        await server.start_and_wait()

    except (KeyboardInterrupt, asyncio.CancelledError):

        # Prints information.
        print("\n[!] Keyboard interrupt detected (Ctrl+C). Closing server ...")
    
    finally:

        # Close server.
        await server.stop()

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    try:
        # Run main with asyncio.
        asyncio.run(main())
    except KeyboardInterrupt:
        pass