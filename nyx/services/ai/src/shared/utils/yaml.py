# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
from typing import Union, Dict, List, Any
# External:
import yaml


# ==============================
# VARIABLES
# ==============================

# Contains valid extensions for yaml functions.
EXTENSIONS:List = ['.yaml', '.yml']


# ==============================
# FUNCTIONS
# ==============================

def load_file(path:Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its content as a dictionary.

    This function validates that the files exists, is not a directory,
    and has a `.yaml` or `.yml` extension. It then safely loads the YAML
    content int a Python dictionary.

    Args:
        path (Union[str, Path]): Path to the YAML file to load.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path points to a directory or is not a `.yaml` or `.yml`
            file.
        yaml.YAMLError: If the YAML content is invalid or cannot be parsed.
    
    Returns:
        Dict[str, Any]: Parsed YAML content as a dictionary.
    """

    # Generates path instance to manage access.
    file_path:Path = Path(path)

    # Checks if the file doesn't exist.
    if not file_path.exists(): raise FileNotFoundError(f"Unable to find file at '{file_path}'")
    # Checks if the file it's valid.
    if file_path.is_dir(): raise ValueError(f"The given path '{file_path}' is a directory, expected {EXTENSIONS} file")
    if file_path.suffix not in ['.yaml', '.yml']: raise ValueError(f"The given file '{file_path}' has invalid format, expected {EXTENSIONS} file")

    # Open the file.
    with file_path.open(mode='r') as file:
        # Load data from yaml file.
        data = yaml.safe_load(stream=file)

    # Checks if the data is a valid dictionary.
    if not isinstance(data, dict): raise ValueError(f"YAML file '{file_path}' did not return a dictionary, got {type(data).__name__}")

    return data