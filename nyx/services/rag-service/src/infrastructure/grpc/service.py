# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from typing import Dict, List, Optional

# Internal:
from application.services.worker_service import WorkerService
from infrastructure.ai_service.client.base import AIServiceAsyncClient
from infrastructure.qdrant.client.base import QdrantAsyncClient
import infrastructure.grpc.generated.rag_service_pb2 as pb2
import infrastructure.grpc.generated.rag_service_pb2_grpc as pb2_grpc


# ==============================
# CLASSES
# ==============================

class RagServiceServicer(pb2_grpc.RAGServiceServicer):
    """
    """

    # ---- Default ---- #

    def __init__(
        self,
        worker_service:WorkerService,
        ai_client:AIServiceAsyncClient,
        qdrant_client:QdrantAsyncClient
    ) -> None:
        """

        """

        # Initializes the class properties.
        self._worker_service:WorkerService = worker_service
        self._ai_client:AIServiceAsyncClient = ai_client
        self._qdrant_client:QdrantAsyncClient = qdrant_client


    # ---- Methods ---- #