# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard: 
import asyncio
# External:
from fastapi import WebSocketDisconnect
from websockets import ConnectionClosedOK
# Internal:
from core.events.bus import EventBus
from core.interfaces.transport import IWebSocketConnection, IReceiverLoop


# ==============================
# CLASSES
# ==============================

class ReceiveLoop(IReceiverLoop):
    """
    Asynchronous loop for receiving WebSocket messages.

    Continuously reads from the WebSocket and publishes received
    messages to the EventBus. Respects SRP by delegating
    all message processing to subscribers.
    """

    # ---- Default ---- #

    def __init__(
        self,
        websocket:IWebSocketConnection,
        event_bus:EventBus
    ) -> None:
        """
        Initialize the receiver loop.

        Args:
            websocket (IWebSocketConnection): WebSocket connection interface.
            event_bus (EventBus): Event bus for publishing messages.
        """
        # Initialize the properties.
        self._websocket:IWebSocketConnection = websocket
        self._event_bus:EventBus = event_bus

        self._task:asyncio.Task|None = None

        self._is_running:bool = False
    

    # ---- Methods ---- #

    async def _run(self) -> None:
        """
        Main receiver loop.

        Continuously receives messages and publishes them to the EventBus.
        Handles connection errors and cancellation gracefully.
        """
        # Try-Except to manage errors.
        try:
            # Main loop.
            while self._is_running:
                # Try-Except to manage errors.
                try:
                    # Gets the message from the WebSocket.
                    message:str = await self._websocket.receive()

                    # Publish event to EventBus.
                    await self._event_bus.publish(
                        event="ws.received",
                        payload=message
                    )

                # If the websocket is closed.
                except (WebSocketDisconnect, ConnectionClosedOK):
                    # Prints information.
                    print("Websocket closed.")
                    # Notify closed connection.
                    await self._event_bus.publish(
                        event="ws.close"
                    )
                    # Ends loop.
                    break
                
                # If the task is cancelled.
                except asyncio.CancelledError:
                    # Ends loop.
                    break

                # If an unexpected error ocurred.
                except Exception as ex:
                    # Publish error.
                    await self._event_bus.publish(
                        event="ws.error",
                        payload=str(ex)
                    )
                    # Awaits 1 second to avoid tight error loop.
                    await asyncio.sleep(delay=1)
            
        # If the task is cancelled.
        except asyncio.CancelledError:
            pass
        
        # Executes finally.
        finally:
            self._is_running = False
            
    async def start(self) -> None:
        """
        Start the receiver loop in a background task.
        """
        # Checks if it's already running.
        if self._is_running: return

        # Creates the task.
        self._is_running = True
        self._task = asyncio.create_task(self._run())
    
    async def stop(self) -> None:
        """
        Stop the receiver loop gracefully.
        """
        # Checks if it's not running.
        if not self._is_running: return

        # Stops the loop.
        self._is_running = False
        # Cancel the task.
        if self._task:
            self._task.cancel()
            # Awaits for the task to finish.
            # with suppress(asyncio.CancelledError):
            await self._task
