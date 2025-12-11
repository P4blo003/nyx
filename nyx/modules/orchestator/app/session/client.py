# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# Internal:
from core.events.bus import EventBus
from core.interfaces.controller import IController
from core.interfaces.transport import IWebSocketConnection, IReceiverLoop, ISenderLoop
from transport.receiver.loop import ReceiveLoop
from transport.sender.loop import SenderLoop


# ==============================
# CLASSES
# ==============================

class ClientSession:
    """
    Orchestrates the WebSocket client lifecycle.

    Responsabilities:
        - Component instantiation and dependency injection.
        - Lifecyclie managment (start/stop).
        - Clean shutdown.
    """

    # ---- Default ---- #

    def __init__(
        self,
        websocket:IWebSocketConnection,
        heartbeat_interval_seconds:float = 30.0
    ) -> None:
        """
        Initializes the client session.
        
        Args:
            websocket (IWebSocketConnection): WebSocket connection implementation.
            heartbeat_interval_seconds (float): Seconds between heartbeats.
        """
        # Intialize the properties.
        self._websocket:IWebSocketConnection = websocket
        self._heartbeat_interval_seconds:float = heartbeat_interval_seconds

        self._event_bus:EventBus|None = None

        self._receiver:IReceiverLoop|None = None
        self._sender:ISenderLoop|None = None

        self._controllers:List[IController] = []

        self._initialized:bool = False
    

    # ---- Methods ---- #

    async def initialize(self) -> None:
        """
        Initialize all components and wire dependencies.
        """
        # Checks if it's already initialized.
        if self._initialized: return

        # Creates the event bus.
        self._event_bus = EventBus()

        # Create transport layer components.
        self._receiver = ReceiveLoop(
            websocket=self._websocket,
            event_bus=self._event_bus
        )
        self._sender = SenderLoop(
            websocket=self._websocket,
            event_bus=self._event_bus
        )

        self._initialized = True
    
    async def start(self) -> None:
        """
        Start all async components.

        This starts the transport layer and heartbeat manager. Controllers
        already listening to events after initialization.
        """
        # Checks if it's not initialized.
        if not self._initialized: return

        # Starts transport layer.
        if self._receiver is not None: await self._receiver.start()
        if self._sender is not None: await self._sender.start()
    
    async def stop(self) -> None:
        """
        Stop all components gracefully.

        Ensures proper cleanup and cancellation off all async tasks.
        """
        # Stops in reverse order of start.
        if self._sender is not None: await self._sender.stop()
        if self._receiver is not None: await self._receiver.stop()

        # Cleanup controllers.
        for controller in self._controllers: await controller.cleanup()

        # Close websocket connection.
        await self._websocket.close()