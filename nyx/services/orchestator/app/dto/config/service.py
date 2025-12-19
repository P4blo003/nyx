# ==========================================================================================
# Author: Pablo González García.
# Created: 18/12/2025
# Last edited: 18/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel


# ==============================
# CLASSES
# ==============================

class ServiceConfig(BaseModel):
    """
    Service configuration.

    Attributes:
        host (str): Service host.
        port (int): Service port.
        ws_ping_interval (int):
        ws_ping_timeout (int):
    """
    # ---- Attributes ---- #

    host:str
    port:int
    ws_ping_interval:int
    ws_ping_timeout:int