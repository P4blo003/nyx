# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from logging import Logger
# Internal:
from core.events.bus import EventBus
from core.config import logger


# ==============================
# CONSTANTS
# ==============================

GLOBAL_EVENT_BUS:EventBus = EventBus()
LOGGER:Logger = logger.get_logger(name="app")