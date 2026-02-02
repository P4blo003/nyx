# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
from typing import Any
from typing import Dict, List
from typing import Union

# External:
import yaml


# ==============================
# FUNCTIONS
# ==============================

SUPPORTED_EXTENSIONS:List[str] = [".yaml", ".yml"]
"""
List of supported YAML file extensions.

These extensions are used to validate input files and ensure that only YAML-formatted
configuration or data are processed by the system.
"""


# ==============================
# FUNCTIONS
# ==============================

def is_yaml(path:Path) -> bool:
    """
    Check whether the given path points to a YAML file.

    The check is based solely on the file extension and does not
    validate the file content.

    Args:
        path (Path): Path object representing the file to check.

    Returns:
        response (bool): `True` if the file haas a supported YAML extension,
            `False` otherwise.
    """

    # Compare the file suffix against the list of allowed YAML extensions.
    return path.suffix in SUPPORTED_EXTENSIONS 

def load(path:Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a YAML file from the given filesystem path.

    This function validates that the provided path exists and points to a file
    (not a directory), then safely loads its YAML contents into a dictionary.

    Args:
        path (Union[str, Path]): Path to the YAML file to be loaded.

    Returns:
        response (Dict[str, Any]): A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the given path does not exists.
        IsADirectoryError: If the given path points to a directory instead of a file.
        ValueError: If the file does not have a supported YAML extension.
        yaml.YAMLError: If the file contents cannot be parsed as valid YAML.
    """

    # Generates Path instance to manage file existence and type.
    f:Path = Path(path)

    # Validates path.
    if not f.exists(): raise FileNotFoundError(f"Unable to find path: {f}")
    if f.is_dir(): raise IsADirectoryError(f"The given path '{f}' is a directory not a file.")
    if not is_yaml(path=f): raise ValueError(
        f"Unsupported file extension '{f.suffix}'"
        f"Expected one of: {SUPPORTED_EXTENSIONS}"
    )

    # Opens the file and loads the data.
    with f.open(mode='r') as stream:
        return yaml.safe_load(stream)