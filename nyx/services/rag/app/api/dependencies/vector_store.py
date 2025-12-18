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
# External:
import yaml
from qdrant_client import AsyncQdrantClient
# Internal:
from core.interfaces.vector_store import IVectorStore
from core.interfaces.vector_store import IVectorStoreController
from core.config.vector_store import StoreConfig, StoreSecretsConfig, StoreCollectionConfig, StoreConnectionConfig
from core.controllers.vector_store import QdrantVectorStoreController
from transport.vector_store.qdrant import QdrantVectorStore



# ==============================
# CONSTANTS
# ==============================

# Vector store.
VECTOR_STORE:IVectorStoreController|None = None
# Vector store configuration.
STORE_CONFIG:StoreConfig|None = None


# ==============================
# FUNCTIONS
# ==============================

def setup(config_path:str|Path) -> None:
    """
    Initializes all vector store dependencies.

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

    # Initializes configuration objects.
    STORE_CONFIG = StoreConfig(
        name=vdb_data["store"]["name"],
        connection=StoreConnectionConfig(**vdb_data["store"]["connection"]),
        collection=StoreCollectionConfig(**vdb_data["store"]["collection"]),
        secrets=StoreSecretsConfig()                                                # type:ignore
    )

    # Match case for vector store.
    match STORE_CONFIG.name:
        # Qdrant vector store.
        case "qdrant":
            # Initializes the vector store transport.
            vs:IVectorStore = QdrantVectorStore(
                client=AsyncQdrantClient(
                    url=f"http://{STORE_CONFIG.connection.host}",
                    port=STORE_CONFIG.connection.http_port,
                    grpc_port=STORE_CONFIG.connection.grpc_port,
                    api_key=str(STORE_CONFIG.secrets.api_key),
                ),
                collection=STORE_CONFIG.collection.name
            )
            # Initializes the vector store controller.
            VECTOR_STORE = QdrantVectorStoreController(vector_store=vs)

async def cleanup() -> None:
    """
    Free resources of all vector store dependencies.
    """
    # Global properties.
    global VECTOR_STORE, STORE_CONFIG

    if VECTOR_STORE: await VECTOR_STORE.cleanup()