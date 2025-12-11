# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
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

class IController(ABC):
    """
    Abstract base class for feature controllers.

    Controllers encapsultae business logic and comunicate
    exclusively through the EventBus.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the controller.

        This method should subscribe to relevant events on the
        EventBus.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup controller resources.

        Called during system shutdown.
        """
        pass