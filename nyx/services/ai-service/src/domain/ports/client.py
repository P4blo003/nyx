# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar, Generic


# ==============================
# TYPES
# ==============================

T = TypeVar("T", bound="IAsyncClient")


# ==============================
# INTERFACES
# ==============================

class ISyncClient(ABC):
    """
    Base interface for any sync client.
    """

    # ---- Methods ---- #

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...


class IAsyncClient(ABC):
    """
    Base interface for any async client.
    """

    # ---- Methods ---- #

    @abstractmethod
    def get_server_url(self) -> str: ...
    
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...


class ISyncClientService(ABC):
    """
    
    """

    # ---- Default ---- #

    @abstractmethod
    def get_client(self, key:str) -> Optional[ISyncClient]: ...

    @abstractmethod
    def startup(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class IAsyncClientService(ABC, Generic[T]):
    """
    
    """

    # ---- Default ---- #
    @abstractmethod
    def get_client(self, key:str) -> Optional[T]: ...

    @abstractmethod
    async def startup(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...