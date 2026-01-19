# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2025
# Last edited: 15/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager

# External:
from fastapi import FastAPI
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from application.config import triton_config


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    FastAPI application lifespan.

    Args:
        app (FastAPI): FastAPI application.
    """

    # Try-Except to manage errors.
    try:

        # Initializes dictionary to keep Triton clients.
        app.state.triton_clients = {}

        # Try-Except to manage errors.
        try:

            # Iterate over triton configuration.
            for key, cfg in triton_config.items():
                # Create gRPC client for Triton container.
                app.state.triton_clients[key] = InferenceServerClient(
                    url=f"{cfg['host']}:{cfg['grpc_port']}",
                    verbose=False
                )
        
        # If an error occurs.
        except Exception as ex:

            # Prints error.
            print(ex)

        yield


        # Free resources.

    # If an error occurs.
    except Exception as ex: raise ex