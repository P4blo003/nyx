# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
from typing import Any, Dict

# External:
import yaml


# ==============================
# FUNCTIONS
# ==============================

def load_data(path:str|Path) -> Dict[str, Any]:
    """"""

    # Generates Path instance to manage file existence and type.
    file_path:Path = Path(path)

    # Checks if the path doesn't exists or if it is not a file.
    if not file_path.exists(): raise FileNotFoundError(f"Unable to find path: {file_path}")
    if file_path.is_dir(): raise IsADirectoryError(f"The given path '{file_path}' is a directory not a file.")

    # Opens the file and loads the data.
    with file_path.open(mode='r') as f:
        return yaml.safe_load(f)