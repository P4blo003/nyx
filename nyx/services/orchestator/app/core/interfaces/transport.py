# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod


# ==============================
# INTERFACES
# ==============================

class IWebSocketConnection(ABC):
    """
    Abstract interface for WebSocket connection operations.

    This interface allows the transport layer to be independent
    of the specific WebSocket library implementation.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def send(self, message:str) -> None:
        """
        Send a message through the WebSocket.

        Args:
            message (str): The message to send (JSON string or text).
        """
        pass

    @abstractmethod
    async def receive(self) -> str:
        """
        Receive the next message from the WebSocket.

        Returns:
            The received message as a string.

        Raises:
            ConnectionClosed: When the connection is closed.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the WebSocket connection gracefully.
        """
        pass

class IReceiverLoop(ABC):
    """
    Interface for the WebSocket message receiver component.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def start(self) -> None:
        """
        Start the receiver loop.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the receiver loop gracefully.
        """
        pass

class ISenderLoop(ABC):
    """
    Interface for the WebSocket message sender component.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def start(self) -> None:
        """
        Start the sender loop.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the sender loop gracefully.
        """
        pass

    @abstractmethod
    async def enqueue_message(self, message:str) -> None:
        """
        Enqueue a message for sending.

        Args:
            message (str): The message to send.
        """
        pass

class IHeartbeatManager(ABC):
    """
    Interface for heartbeat management.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def start(self) -> None:
        """
        Start the heartbeat manager.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the heartbeat manager gracefully.
        """
        pass