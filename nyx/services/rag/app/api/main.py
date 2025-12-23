# ==========================================================================================
# Author: Pablo González García.
# Created: 23/12/2025
# Last edited: 23/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import FastAPI


# ==============================
# CONSTANTS
# ==============================

__APP:FastAPI|None = None


# ==============================
# FUNCTIONS
# ==============================

def setup() -> FastAPI:
    """
    Initializes `FastAPI` application.
    """

    # Global variables.
    global __APP

    # Initializes fastapi application.
    __APP = FastAPI()

    return __APP