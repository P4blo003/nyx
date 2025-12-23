# ==========================================================================================
# Author: Pablo González García.
# Created: 23/12/2025
# Last edited: 23/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
# External:
import dotenv


# ==============================
# FUNCTIONS
# ==============================

def load_env(file:str|Path|None = None) -> None:
    """
    Load configuration from `file` if it's not None or from
    default `.env` file.

    Args:
        file (str|Path|None): Environment variables file.
    """
    
    # Creates Path instance to manage access.
    path:Path = Path(file) if file is not None else Path(".env")

    # Loads environment variables.
    dotenv.load_dotenv(path)