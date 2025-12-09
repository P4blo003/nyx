# ==========================================================================================
# Author: Pablo González García.
# Created: 09/12/2025
# Last edited: 09/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import uuid
import asyncio
from enum import Enum
from typing import Optional, Dict
from contextlib import suppress
# External:
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
# Internal:
from schemas.web.base import MessageType
from schemas.web.heartbeat import HeartbeatResponse, HeartbeatContent


# ==============================
# ENUMS
# ==============================

class HeartbeatStatus(Enum):
    """
    Represents the operational state of the heartbeat manager.
    """
    # ---- Attributes ---- #

    ACTIVE = 'active'
    WAITING = 'wainting'
    MISSING = 'missing'
    FAILED = 'failed'
    STOPPED = 'stopped'


# ==============================
# CLASSES
# ==============================

class HeartbeatManager:
    """
    Manages sending heartbeats messages from server to client over
    websockets.
    """
    # ---- Default ---- #

    def __init__(
        self,
        websocket:WebSocket,
        interval_seconds:float = 15.0,
        timeout_seconds:float = 30.0,
        max_missing_heartbeats:int = 3
    ) -> None:
        """
        Initialize the heartbeat manager.

        Args:
            websocket (WebSocket): The WebSocket transport to send message.
            interval_seconds (float): Interval between heartbeat messages.
            timeout_seconds (float): Maximum wait time for heartbeat response.
            max_missing_heartbeats (int): Number of missed heartbats before
                failure.
        """
        # Intialize the properties.
        self.websocket:WebSocket = websocket
        self.interval_seconds:float = interval_seconds
        self.timeout_seconds:float = timeout_seconds
        self.max_missing_heartbeats:int = max_missing_heartbeats

        self.heartbeats_sent:int = 0
        self.heartbeats_received:int = 0
        self.missing_heartbeats:int = 0

        self._current_heartbeat_id:Optional[str] = None
        self._response_event:asyncio.Event = asyncio.Event()
        self._task:Optional[asyncio.Task] = None

        self.status:HeartbeatStatus = HeartbeatStatus.STOPPED


    # ---- Methods ---- #

    async def _send_heartbeat(self) -> bool:
        """
        Send a heartbeat message to the client.

        Returns:
            bool: True if successfully sent, False if an error ocurred.
        """
        # Try-Except to manage errors.
        try:
            # Generate the heartbeat id.
            self._current_heartbeat_id = str(uuid.uuid4)
            # Clear the response event.
            self._response_event.clear()

            # Generates the payload.
            payload:Dict = HeartbeatResponse(
                type=MessageType.HEARTBEAT,
                timestamp=asyncio.get_running_loop().time(),
                content=HeartbeatContent(id=self._current_heartbeat_id)
            ).model_dump(by_alias=True)

            # Send the heartbeat.
            await self.websocket.send_json(data=payload)
            # Update metrics.
            self.heartbeats_sent += 1
            
            return True

        # If the webscokect is disconnected.
        except (WebSocketDisconnect, RuntimeError):
            # Prints error.
            print(f"Unable to send heartbeat. WebSocket is disconnected.")
            # Sets the status.
            self.status = HeartbeatStatus.FAILED

            return False
        
        # If an unexpected error ocurred.
        except Exception as ex:
            # Prints error.
            print(f"Error during heartbeat shipment: {ex}")
            # Sets the status.
            self.status = HeartbeatStatus.FAILED

            return False
    
    async def _await_for_response(self) -> bool:
        """
        Wait for heartbeat acknowledgement from the client.

        Returns:
            bool: True if the response was received in time, False otherwise.
        """
        # Try-Except to manage errors.
        try:
            # Awaits for the response.
            await asyncio.wait_for(
                self._response_event.wait(),
                timeout=self.timeout_seconds
            )

            return True

        # If the timeout end.
        except asyncio.TimeoutError:
            # Prints information.
            print(f"Timeout.")

            return False
        
    async def _loop(self) -> None:
        """
        Main loop for sending heartbeats and handling responses.
        """
        # Sets intial value.
        self.missing_heartbeats = 0

        # Try-Except to manage errors.
        try:
            # Main loop while status is not stopped o failed.
            while self.status != HeartbeatStatus.STOPPED and self.status != HeartbeatStatus.FAILED :

                # Sleep.
                await asyncio.sleep(delay=self.interval_seconds)

                # Checks if it's active.
                if self.status != HeartbeatStatus.ACTIVE: break

                # Sends the heartbeat.
                if not await self._send_heartbeat(): break

                # Await for response.
                receive:bool = await self._await_for_response()

                # Checks if the heartbeat was received.
                if receive:
                    # Update metrics.
                    self.missing_heartbeats = 0
                    self.heartbeats_received += 1
                # If the heartbeat was not received.
                else:
                    # Update metrics.
                    self.missing_heartbeats += 1
                    self.status = HeartbeatStatus.MISSING
                    
                    # Checks if the limit has been reached.
                    if self.missing_heartbeats > self.max_missing_heartbeats:
                        # Prints information.
                        print(f"Hearbeat failure. Max num of missing heartbeats reached.")
                        # Update metrics.
                        self.status = HeartbeatStatus.FAILED
                        break
        
        # If the task is cancelled.
        except asyncio.CancelledError:
            raise

        # If an unexpected error ocurred.
        except Exception as ex:
            # Prints information.
            print(f"Unexpected error during main loop: {ex}")
            # Update metrics.
            self.status = HeartbeatStatus.FAILED
        
    async def start(self) -> None:
        """
        Start the heartbeat manager loop.

        Raises:
            RuntimeError: If the heartbeat loop is already running.
        """
        # Checks if the manager is already running.
        if self.status != HeartbeatStatus.STOPPED and self.status != HeartbeatStatus.FAILED:
            raise RuntimeError("HeartbeatManager is already running.")

        # Update metrics.
        self.status = HeartbeatStatus.ACTIVE
        # Initialies the task.
        self._task = asyncio.create_task(
            self._loop(),
            name=f"hb-{uuid.uuid4().hex[:8]}"
        )

    async def notify_response(
        self,
        heartbeat_id:str
    ) -> bool:
        """
        Notify the manager that a heartbeat response has been received.

        Args:
            heartbeat_id (str): The identifier of the received heartbeat.

        Returns:
            bool: True if the heartbeat matches the current one, False otherwise
        """
        # Checks if the heartbeat id has been initialied and is equal to the given one.
        if self._current_heartbeat_id is None or self._current_heartbeat_id != heartbeat_id: return False

        # Notify event.
        self._response_event.set()

        return True   
    
    async def stop(self) -> None:
        """
        Stop the heartbeat manager and cancel the running loop task.
        """
        # Update metrics.
        self.status = HeartbeatStatus.STOPPED

        # Checks if the task is not None and is not done.
        if self._task is not None and not self._task.done():
            # Cancel task.
            self._task.cancel()
            # Stops the task.
            with suppress(asyncio.CancelledError): await self._task

        # Sets to None.
        self._task = None