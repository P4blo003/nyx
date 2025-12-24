# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
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
    Loads environment variables from `file` or try to load
    from default `.env`.
    """
    
    # Creates Path instance to manage access.
    path:Path = Path(file) if file is not None else Path(".env")

    # Load environment variables.
    dotenv.load_dotenv(dotenv_path=path)