# ==========================================================================================
# Author: Pablo González García.
# Created: 23/12/2025
# Last edited: 23/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    
    """
    yield