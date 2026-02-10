# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 09/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
import time
import logging
from typing import Dict, List, Optional
from urllib import request

# External:
import grpc
from grpc import aio
import numpy as np
from tritonclient.grpc.aio import InferInput, InferRequestedOutput, InferResult

# Internal:
from domain.ports.client import IAsyncClient, IAsyncClientService
import infrastructure.grpc.generated.ai_service_pb2 as pb2
import infrastructure.grpc.generated.ai_service_pb2_grpc as pb2_grpc
from infrastructure.triton.client.base import TritonAsyncClient
from infrastructure.triton.config.task import TritonTask


# ==============================
# CLASSES
# ==============================

class AIServiceServicer(pb2_grpc.AIServiceServicer):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        client_service:IAsyncClientService[TritonAsyncClient],
        tasks:Dict[str, TritonTask]
        
    ) -> None:
        """
        
        """

        # Initializes the class properties.
        self._client_service = client_service
        self._tasks:Dict[str, TritonTask] = tasks

        self._log:logging.Logger = logging.getLogger("AIServiceServicer")


    # ---- Methods ---- #

    async def load_model(
        self,
        request:pb2.LoadModelRequest,
        context
    ):
        """
        
        """

        try:
            client:Optional[TritonAsyncClient] = self._client_service.get_client(key=request.server)

            if client is None: raise ValueError(f"Client for server '{request.server}' not found.")
            await client.load_model(model_name=request.name, model_version=request.version.strip() or "")

            return pb2.ModelStatus(
                name=request.name,
                version=request.version or "",
                server=request.server,
                state=pb2.ModelStatus.READY,
                message="Model loaded successfully"
            )

        except Exception as e:

            self._log.error(f"Error loading model '{request.name}:{request.version}' on server '{request.server}': {str(e)}")
            context.set_details(str(e))
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        
    async def unload_model(
        self,
        request:pb2.UnloadModelRequest,
        context
    ):
        """
        """

        try:
            client:Optional[TritonAsyncClient] = self._client_service.get_client(key=request.server)

            if client is None: raise ValueError(f"Client for server '{request.server}' not found.")
            await client.unload_model(model_name=request.name)

            return pb2.ModelStatus(
                name=request.name,
                version=request.version or "",
                server=request.server,
                state=pb2.ModelStatus.UNLOADED,
                message="Model unloaded successfully"
            )
        
        except Exception as e:

            self._log.error(f"Error unloading model '{request.name}' on server '{request.server}': {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def make_infer(
        self,
        request:pb2.InferenceRequest,
        context
    ):
        """
        
        """

        try:

            if request.HasField("text_batch"):
                inputs: List[bytes] = [text.encode("utf-8") for text in request.text_batch.content]
                name = "TEXT"

            elif request.HasField("image_batch"):
                inputs: List[bytes] = list(request.image_batch.content)
                name = "IMAGE"

            else:
                raise ValueError("InferenceRequest contains an empty batch")
            

            task:Optional[TritonTask] = self._tasks[request.task]
            if task is None: raise ValueError(f"No task configured for '{request.task}'")
    
            server:Optional[str] = self._tasks[request.task].endpoint
            if server is None: raise ValueError(f"Endpoint for task '{request.task}' not found.")

            client:Optional[TritonAsyncClient] = self._client_service.get_client(key=server)
            if client is None: raise ValueError(f"Client for server '{server}' not found.")
    
            input_data:np.ndarray = np.array(inputs, dtype=object).reshape([len(inputs), 1])
            
            infer_input:InferInput = InferInput(
                name=name,
                shape=input_data.shape,
                datatype="BYTES"
            )
            infer_input.set_data_from_numpy(input_data)
            
            outputs:InferRequestedOutput = InferRequestedOutput("EMBEDDING")

            response:InferResult = await client.make_infer(
                model_name=self._tasks[request.task].model_name,
                model_version="",
                input_data=[infer_input],
                output_data=[outputs]
            )

            embeddings:Optional[np.ndarray] = response.as_numpy("EMBEDDING")
            embedding_batch:pb2.EmbeddingBatch = pb2.EmbeddingBatch()

            if embeddings is None:
                raise RuntimeError("Triton returned no EMBEDDING output")

            if embeddings.ndim != 2:
                raise RuntimeError("Unexpected embedding tensor shape")

            for row in embeddings:
                embedding_batch.vectors.add().values.extend(
                    float(v) for v in row
                )

            result:pb2.InferenceResponse = pb2.InferenceResponse(task=request.task)
            result.embedding_batch.CopyFrom(embedding_batch)

            return result

        except Exception as e:

            self._log.error(f"Error making inference for task '{request.task}': {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))