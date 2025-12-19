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
from dto.config.service import ServiceConfig
from utilities import yaml


# ==============================
# CONSTANTS
# ==============================

SERVICE_CONFIG:ServiceConfig|None = None


# ==============================
# FUNCTIONS
# ==============================

def setup_settings(config_path:str|Path) -> None:
    """
    Configure settings from config file.

    Args:
        config_path (str|Path): Config file path.
    """
    # Global variables.
    global SERVICE_CONFIG

    # Creates Path instance to manage access.
    config_path = Path(config_path)

    # Initialize the configurations.
    SERVICE_CONFIG = ServiceConfig(**yaml.load(file_path=config_path.joinpath("service.yml")))