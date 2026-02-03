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

# Internal:
from domain.ports.client import ISyncClient, IAsyncClient


# ==============================
# INTERFACES
# ==============================

class TritonSyncClient(ISyncClient, ABC):
    """
    Abstract base class for Triton sync clients.
    Adds Triton-specific API on top of ``ISyncClient``.
    """

    # ---- Methods ---- #

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

class TritonAsyncClient(IAsyncClient, ABC):
    """
    Abstract base class for Triton async clients.
    Adds Triton-specific API on top of ``IAsyncClient``.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...