# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
# External:
import yaml
# Internal:
from core.interfaces.vector_store import IVectorStore
from core.config.vector_store import StoreConfig, StoreSecretsConfig, StoreCollectionConfig, StoreConnectionConfig


# ==============================
# CONSTANTS
# ==============================

# Vector store.
VECTOR_STORE:IVectorStore|None = None
# Vector store configuration.
STORE_CONFIG:StoreConfig|None = None


# ==============================
# FUNCTIONS
# ==============================

def setup_dependencies(config_path:str|Path) -> None:
    """
    Initializes all dependencies.

    Args:
        config_path (str|Path): Config directory.
    """
    # Global properties.
    global VECTOR_STORE, STORE_CONFIG

    # Creates Path instance to manage access.
    config_path = Path(config_path)

    # Loads configurations.
    with open(Path(config_path.joinpath("vdb.yml")), "r") as file:
        vdb_data = yaml.safe_load(file)

    # Initialize the instances.
    STORE_CONFIG = StoreConfig(
        name=vdb_data["store"]["name"],
        connection=StoreConnectionConfig(**vdb_data["store"]["connection"]),
        secrets=StoreSecretsConfig(),       # type: ignore
        collection=StoreCollectionConfig(**vdb_data["store"]["collection"])
    )

    pass