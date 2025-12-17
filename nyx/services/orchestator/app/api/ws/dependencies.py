# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
# Internal:
from core.events.bus import EventBus


# ==============================
# CONSTANTS
# ==============================

GLOBAL_EVENT_BUS:EventBus|None = None


# ==============================
# FUNCTIONS
# ==============================

def setup_dependencies(config_path:str|Path) -> None:
    """
    Initialize all dependencies.

    Args:
        config_path (str|Path): Config directory.
    """
    # Global properties.
    global GLOBAL_EVENT_BUS

    # Initializes the properties.
    GLOBAL_EVENT_BUS = EventBus()