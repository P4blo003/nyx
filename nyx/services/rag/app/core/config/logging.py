# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging.config
from logging import Logger
from pathlib import Path
# External:
import yaml


# ==============================
# FUNCTIONS
# ==============================

def setup_logging(config_path:str|Path):
    """
    Configure logging system from config file.

    Args:
        config_path (str|Path): Config file path.
    """
    # Creates log directory.
    log_dir:Path = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Creates Path instance to manage access.
    config_path = Path(config_path)

    # Load configuration file.
    with open(config_path.joinpath("logging.yml"), 'r') as f:
        # Load file.
        config = yaml.safe_load(f)

    # Apply configuration.
    logging.config.dictConfig(config)

def get_logger(name:str) -> Logger:
    """
    Gets configured logger.

    Args:
        name (str): Logger's name.

    Returns:
        Logger: Configured logger.
    """
    return logging.getLogger(name=name)