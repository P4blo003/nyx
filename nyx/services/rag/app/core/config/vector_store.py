# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:

# External:
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings



# ==============================
# CLASSES
# ==============================

class StoreConnectionConfig(BaseModel):
    """
    Store connection configuration.
    """
    # ---- Attributes ---- #

    host:str
    http_port:int
    grpc_port:int
    prefer_grpc:bool
    timeout_seconds:int

class StoreSecretsConfig(BaseSettings):
    """
    Store secrets configuration.
    """
    # ---- Attributes ---- #

    api_key:SecretStr


    # ---- Classes ---- #

    class Config:
        env_file = ".env"

class StoreCollectionConfig(BaseModel):
    """
    Store collection configuration.
    """
    # ---- Attributes ---- #

    name:str
    vector_size:int
    distance:str

class StoreConfig(BaseModel):
    """
    Store configuration.
    """
    # ---- Attributes ---- #

    name:str
    connection:StoreConnectionConfig
    secrets:StoreSecretsConfig
    collection:StoreCollectionConfig