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
from typing import Any, Optional

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
    async def load_model(self, model_name:str, model_version:Optional[str]) -> None: ...

    @abstractmethod
    async def unload_model(self, model_name:str, unload_dependents:bool = True) -> None: ...

    @abstractmethod
    async def make_infer(self, model_name:str, model_version:Optional[str], input_data:Any, output_data:Any) -> Any: ...
