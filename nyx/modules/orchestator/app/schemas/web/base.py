# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 04/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
from enum import Enum


# ==============================
# ENUMS
# ==============================

class MessageType(str, Enum):
    """
    Enumerate the different message categories
    available in the system.
    """
    # ---- Attributes ---- #
    HEARTBEAT = 'heartbeat'