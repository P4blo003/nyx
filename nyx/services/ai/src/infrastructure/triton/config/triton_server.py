# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 22/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class TritonInferenceServer(BaseModel):
    """
    Represents the configuration and connection details for a single Triton Inference
    Server.

    This model holds the network information required to establish a connection via
    HTTP or gRPC to a deployed Triton Inference Server.

    Attributes:
        host (str): The hostname or IP address where the Triton Inference Server
            is running.
        http_port (int): The port number exposed for REST API inference requests.
        grpc_port (int): The port number exposed for gRPC inference requests.
    """

    # ---- Attributes ---- #

    host:str = Field(..., description="The hostname or IP address where the Triton Inference Server is running")
    http_port:int = Field(..., description="The port number exposed for REST API inference requests")
    grpc_port:int = Field(..., description="The port number exposed for gRPC inference requests")

class TritonInferenceServerMapping(BaseModel):
    """
    A registry mapping unique identifiers to their corresponding Triton Inference Server.

    This container allows managing multiple Triton instances, enabling lookups
    by specific string key.

    Attributes:
        servers (Dict[str, TritonInferenceServer]): A dictionary where the key is the
            identifier string and the value is the Triton Inference Server configuration.
    """

    # ---- Attributes ---- #

    servers:Dict[str, TritonInferenceServer] = Field(..., description="A dictionary mapping unique server identifiers (keys) to TritonInferenceServer instances (values).")