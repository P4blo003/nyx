# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
import sys
# External:
from fastapi import APIRouter, WebSocket
from websockets import ConnectionClosedOK
# Internal:
from session.client import ClientSession
from transport.websocket.adapter import FastApiWebSocketAdapter
from api.ws.dependencies import GLOBAL_EVENT_BUS as global_event_bus


# ==============================
# CONSTANTS
# ==============================

# Try-Except to manage errors.
try:
    # Initilize the router.
    router:APIRouter = APIRouter(
        prefix="/ws",
        tags=["WebSocket"]
    )

# If an unexpected error ocurred.
except Exception as ex:
    # Prints information.
    print(f"Fatal error at {__name__}: {ex}")
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
    and manages the lifecylce of the session.

    Args:
        websocker (WebSocket): The incoming WebSocket connection.
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

    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Error handling request: {ex}")

    # Executes finally.
    finally:
        # Close the session.
        await session.stop()