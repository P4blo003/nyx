# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import sys
import logging
# External:
from fastapi import APIRouter, WebSocket
from websockets import ConnectionClosedOK
# Internal:
from session.client import ClientSession
from transport.websocket.adapter import FastApiWebSocketAdapter
from api.ws.dependencies import GLOBAL_EVENT_BUS as global_event_bus
from core.logging.facade import Log


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initialize the router.
    router:APIRouter = APIRouter(
        prefix="/ws",
        tags=["WebSocket"]
    )

# If an unexpected error occurred.
except Exception as ex:
    # Prints information.
    logging.critical("Unable to initialize router", exc_info=True)
    # End the program.
    sys.exit(1000)


# ==============================
# ENDPOINTS
# ==============================

@router.websocket("/chat")
async def chat(
    websocket:WebSocket
) -> None:
    """
    WebSocket endpoint for chat functionality.

    This endpoint accepts WebSocket connections, initializes a client session,
    and manages the lifecycle of the session.

    Args:
        websocket (WebSocket): The incoming WebSocket connection.
    """
    # Awaits for websocket connection.
    await websocket.accept()

    # Initialize the WebSocket adapter and session.
    adapter:FastApiWebSocketAdapter = FastApiWebSocketAdapter(websocket=websocket)
    session:ClientSession = ClientSession(
        websocket=adapter,
        global_event_bus=global_event_bus
    )

    # Try-Except to manage errors.
    try:
        # Initializes, starts and awaits session.
        await session.initialize()
        await session.start()
        await session.wait()

    # To avoid show close message.
    except ConnectionClosedOK:
        # Pass
        pass

    # If an unexpected error occurred.
    except Exception as ex:
        # Prints information.
        await Log.error("Error during request processing.", exception=ex)

    # Executes finally.
    finally:
        # Close the session.
        await session.stop()