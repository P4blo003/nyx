# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2025
# Last edited: 15/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from typing import Dict, Any


# ==============================
# CONFIGURATIONS
# ==============================

# Triton configuration.
triton_config:Dict[str, Any] = {
    "vllm": {
        "host": os.environ.get("VLLM_HOST", "localhost"),
        "http_port": os.environ.get("VLLM_HTTP_PORT", "8000"),
        "grpc_port": os.environ.get("VLLM_GRPC_PORT", "8001")
    },
    "onnx": {
        "host": os.environ.get("ONNX_HOST", "localhost"),
        "http_port": os.environ.get("ONNX_HTTP_PORT", "8000"),
        "grpc_port": os.environ.get("ONNX_GRPC_PORT", "8001")
    }
}