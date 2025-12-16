# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import WebSocket
# Internal:
from core.interfaces.transport import IWebSocketConnection


# ==============================
# CLASSES
# ==============================

class FastApiWebSocketAdapter(IWebSocketConnection):
    """
    Adapter for FastAPI WebSocket connections.

    This class implements the IWebSocketConnection interface and provides
    methods to send, receive and close a WebSocket connection using FastAPI's
    WebSocket implementation.
    """

    # ---- Default ---- #

    def __init__(
        self,
        websocket:WebSocket
    ) -> None:
        """
        Initialize the FastAPI WebSocket adapter.

        Args:
            websocket (WebSocket): The FastAPI WebSocket instance.
        """
        # Initialize the properties.
        self._websocket:WebSocket = websocket
    

    # ---- Methods ---- #

    async def send(self, message: str) -> None:
        """
        Send a text message over the WebSocket connection.

        Args:
            message (str): The message to send.
        
        Raises:
            WebSocketException: If sending fails.
        """
        # Awaits to send the message.
        await self._websocket.send_text(data=message)

    async def receive(self) -> str:
        """
        Receive a text message from the WebSocket connection.

        Returns:
            str: The text message received from the client.
        
        Raises:
            WebSocketException: If receiving fails.
        """
        # Awaits to receive the message.
        return await self._websocket.receive_text()

    async def close(self) -> None:
        """
        Close the WebSocket connection.

        Raises:
            WebSocketException: If closing fails.
        """
        # Awaits to close the WebSocket.
        await self._websocket.close()