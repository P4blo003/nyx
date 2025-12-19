# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
from typing import Dict, Any
# External:
import yaml


# ==============================
# FUNCTIONS
# ==============================

def load(file_path:str|Path) -> Dict[str, Any]:
    """
    Load data from yaml file.

    Args:
        file_path (str|Path): File's path.

    Returns:
        Dict[str, Any]: File's content.
    """
    file:Path = Path(file_path)

    # Open the file and load the data.
    with file.open("r") as f:
        return yaml.safe_load(f)
