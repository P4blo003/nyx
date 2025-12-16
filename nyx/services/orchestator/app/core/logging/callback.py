# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any
# Internal:
from core.logging.facade import Log


# ==============================
# FUNCTIONS
# ==============================

async def on_received(_:Any) -> None:
    await Log.info(message="Message received")

async def on_sent(_:Any) -> None:
    await Log.info(message="Message sent")

async def on_error(payload:Any) -> None:
    await Log.info(message=payload)