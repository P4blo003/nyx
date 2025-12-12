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
from typing import List, Dict, Callable, Awaitable, Any


# ==============================
# CLASSES
# ==============================

class EventBus:
    """
    Asynchronous event bus for decoupled component comunication.

    Supports multiple subscribers per event type. All callbacks are
    executed concurrently when an event is published.

    Thread-safe for asyncio operations within a single event loop.
    """

    # ---- Default ---- #

    def __init__(self) -> None:
        """
        Initializes the event bus with an empty subscriber registry.
        """
        # Initializes the properties.
        self._subscribers:Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        self._lock:asyncio.Lock = asyncio.Lock()

    
    # ---- Methods ---- #

    async def _safe_callback(
        self,
        event:str,
        callback:Callable[[Any], Awaitable[None]],
        payload:Any
    ) -> None:
        """
        Execute a callback with error handling.

        Args:
            event (str): Event name.
            callback (Callable[[Any], Awaitable[None]]): The subscriber callback to execute.
            payload (Any): Event payload.
        """
        # Try-Except to manage errors.
        try:
            # Await callback.
            await callback(payload)

        # If an unexpected error ocurred.
        except Exception as ex:
            # Prints information.
            print(f"Error in Callback for event {event}: {ex}")

    async def subscribe(
        self,
        event:str,
        callback:Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Subscribe a callback to an event type.

        Args:
            event (str): The event type to listen for.
            callback (Callable[[Any], Awaitable[None]]): Async function to
                call when event is published.
        """
        # Add the subscriber.
        async with self._lock:
            # Checks if the event is not in subscribers.
            if event not in self._subscribers:
                # Creates the registry.
                self._subscribers[event] = []
            # Add the callback.
            self._subscribers[event].append(callback)
    
    async def unsubscribe(
        self,
        event:str,
        callback:Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Unsubscribe a callback to an event type.

        Args:
            event (str): The event type to listen for.
            callback (Callable[[Any], Awaitable[None]]): Async function to
                call when event is published.
        """
        # Add the subscriber.
        async with self._lock:
            # Checks if the callback is in the list.
            if callback in self._subscribers[event]:
                # Remove the callback.
                self._subscribers[event].remove(callback)
    
    async def publish(
        self,
        event:str,
        payload:Any|None = None
    ) -> None:
        """
        Publish an event to all subscribers.

        All subscribers callbacks are executed concurrently. Exceptions
        in individual callbacks are printed but don't
        affect other callbacks.

        Args:
            event (str): The event type to publish.
            payload (Any): Data to pass to subscribers.
        """
        # Gets all the callbacks that mathc the event.
        async with self._lock:
            callbacks:List[Callable[[Any], Awaitable[None]]] = self._subscribers.get(event, []).copy()

        # Checks if there are callbacks to execute.
        if not callbacks: return

        #Executes all callbacks concurrently.
        tasks:List = [self._safe_callback(
            event=event,
            callback=callback,
            payload=payload
        ) for callback in callbacks]
        await asyncio.gather(*tasks, return_exceptions=True)