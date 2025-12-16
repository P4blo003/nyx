
# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import json
from typing import Dict, Any
# Internal:
from core.events.bus import EventBus
from core.interfaces.controller import IController
from core.logging.facade import Log
from dto.request import ClientRequest


# ==============================
# CLASSES
# ==============================

class OrchestratorController(IController):
    """
    Controller that orchestrates the message flow.

    It subscribes to 'ws.message.received', classifies the message,
    and routes it to the appropriate service.
    """

    # ---- Default ---- #

    def __init__(
        self,
        event_bus: EventBus
    ) -> None:
        """
        Initialize the OrchestratorController.

        Args:
            event_bus (EventBus): The application event bus.
            classifier (IQueryClassifier): The strategy to classify messages.
        """
        self._event_bus:EventBus = event_bus


    # ---- Methods ---- #

    async def _on_message_received(self, payload: Any) -> None:
        """
        Callback for when a WebSocket message is received.

        Args:
            payload (Any): The message payload (expected string).
        """
        # Try-Except to manage errors.
        try:
            # Convert into dictionary.
            data:Dict[str, Any] = json.loads(str(payload))
            # Deserializes the payload into a ClientRequest class.
            request:ClientRequest = ClientRequest(**data)
        
        # If an unexpected error occurred.
        except Exception as ex:
            # Prints information.
            await Log.error(message=f"Error during message handling", exception=ex)

    async def initialize(self) -> None:
        """
        Initialize the controller subscriptions.
        """
        # Subscribe to message received event.
        await self._event_bus.subscribe(
            event="ws.received",
            callback=self._on_message_received
        )

    async def cleanup(self) -> None:
        """
        Cleanup the controller subscriptions.
        """
        # Unsubscribe from message received event.
        await self._event_bus.unsubscribe(
            event="ws.received",
            callback=self._on_message_received
        )