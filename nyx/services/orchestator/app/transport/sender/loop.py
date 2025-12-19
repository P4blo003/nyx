# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from contextlib import suppress
# External:
from fastapi import WebSocketDisconnect
from websockets import ConnectionClosedOK
# Internal:
from core.events.bus import EventBus
from core.interfaces.transport import IWebSocketConnection, ISenderLoop


# ==============================
# CLASSES
# ==============================

class SenderLoop(ISenderLoop):
    """
    Asynchronous loop for sending WebSocket messages.

    Maintains an internal queue for outgoing messages and processes them
    sequentially. Ensures messages are sent in order and handles backpressure
    naturally through queue blocking.
    """

    # ---- Default ---- #

    def __init__(
        self,
        websocket:IWebSocketConnection,
        event_bus:EventBus,
        max_queue_size:int = 1000
    ) -> None:
        """
        Initialize the sender loop.

        Args:
            websocket (IWebSocketConnection): WebSocket connection interface.
            event_bus (EventBus): Event bus for receiving send events.
            max_queue_size (int): Maximum number of queued messages.
        """
        # Initialize the properties.
        self._websocket:IWebSocketConnection = websocket
        self._event_bus:EventBus = event_bus
        self._queue:asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        self._task:asyncio.Task|None = None

        self._is_running:bool = False

    
    # ---- Methods ---- #

    async def _handle_send_message(
        self,
        payload:str
    ) -> None:
        """
        Handle send events from the EventBus.

        Args:
            payload (str): The message to send.
        """
        # Add the message to the queue.
        await self.enqueue_message(message=payload)
    
    async def _run(self) -> None:
        """
        Main sender loop.

        Continuously processes messages from the queue and sends them
        through the WebSocket connection.
        """
        # Try-Except to manage errors.
        try:
            # Main loop.
            while self._is_running:
                # Try-Except to manage errors.
                try:
                    # Wait for next message with timeout to allow clean shutdown.
                    message:str = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                    
                    # Sends the message.
                    await self._websocket.send(message=message)
                
                # If timeout.
                except asyncio.TimeoutError:
                    # No message received, continue loop.
                    continue

                # If the websocket is closed.
                except (WebSocketDisconnect, ConnectionClosedOK):
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
                    
                # If an unexpected error occurred.
                except Exception as ex:
                    # Publish error.
                    await self._event_bus.publish(
                        event="ws.error",
                        payload=str(ex)
                    )
                    
        # If the task is cancelled.
        except asyncio.CancelledError:
            pass
        
        # Executes finally.
        finally:
            self._is_running = False

    async def start(self) -> None:
        """
        Start the sender loop and subscribe to send events.
        """
        # Checks if it's already running.
        if self._is_running: return

        self._is_running = True

        # Subscribe and send events.
        await self._event_bus.subscribe(
            event="ws.sent",
            callback=self._handle_send_message
        )

        # Creates the task.
        self._task = asyncio.create_task(self._run())
    
    async def stop(self) -> None:
        """
        Stop the sender loop gracefully.
        """
        # Checks if it's not running.
        if not self._is_running: return

        # Stops the loop.
        self._is_running = False
        # Cancel the task.
        if self._task:
            self._task.cancel()
            # Awaits for the task to finish.
            with suppress(asyncio.CancelledError):
                await self._task
        
        # Unsubscribe.
        await self._event_bus.unsubscribe(
            event="ws.sent",
            callback=self._handle_send_message
        )
    
    async def enqueue_message(
        self,
        message: str
    ) -> None:
        """
        Enqueue a message for sending.

        Args:
            message (str): The message to send.
        
        Raises:
            asyncio.QueueFull: If the queue is at capacity.
        """
        # Try-Except to manage errors.
        try:
            # Add the message to the queue.
            await self._queue.put(message)
        
        # If the queue is at capacity.
        except asyncio.QueueFull:
            # Raise the error.
            raise
    
