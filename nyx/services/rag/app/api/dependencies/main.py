# ==========================================================================================
# Author: Pablo González García.
# Created: 18/12/2025
# Last edited: 18/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
# Internal:
from api.dependencies import vector_store


# ==============================
# FUNCTIONS
# ==============================

def setup(config_path:str|Path) -> None:
    """
    Initializes all dependencies.

    Args:
        config_path (str): Config directory.
    """
    # Initializes all dependencies.
    vector_store.setup(config_path)

async def cleanup() -> None:
    """
    Free resources of all dependencies.
    """
    # Cleanup all dependencies.
    await vector_store.cleanup()