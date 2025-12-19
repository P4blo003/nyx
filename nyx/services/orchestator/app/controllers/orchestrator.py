
# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
import json
from typing import Dict, List, Any
# Internal:
from core.events.bus import EventBus
from core.interfaces.controller import IController
from core.interfaces.handler import IRagHandler
from core.logging.facade import Log
from dto.request import ClientRequest
from utilities.task import cancel_task


# ==============================
# CLASSES
# ==============================

class OrchestratorController(IController):
    """
    Controller that orchestrates the message flow.

    It subscribes to 'ws.received', classifies the message,
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
        """
        # Initialize properties.
        self._event_bus:EventBus = event_bus

        self._shutting_down:bool = False
        self._active_task:asyncio.Task|None = None

        self._handlers:List[IRagHandler] = []


    # ---- Methods ---- #

    async def _on_message_received(self, payload: Any) -> None:
        """
        Callback for when a WebSocket message is received.

        Args:
            payload (Any): The message payload (expected string).
        """
        # Try-Except to manage errors.
        try:
            # Checks if it's shutting down.
            if self._shutting_down: raise RuntimeError("Unable to handle received message. The orchestrator is shutting down.")

            # Convert into dictionary.
            data:Dict[str, Any] = json.loads(str(payload))
            # Deserializes the payload into a ClientRequest class.
            request:ClientRequest = ClientRequest(**data)

            # TODO: Process message type.

            # TODO: Manage handlers.

            # TODO: Execute task and awaits for results.

            # TODO: Generate prompt.

            # TODO: Send prompt to LLM Service.

            # TODO: Get response from LLM Service in stream and publish into ws.sent event.
        
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
        # Sets it's shutting down.
        self._shutting_down = True

        #Await to cancel the task.
        self._active_task = await cancel_task(task_to_cancel=self._active_task)

        # Unsubscribe from message received event.
        await self._event_bus.unsubscribe(
            event="ws.received",
            callback=self._on_message_received
        )