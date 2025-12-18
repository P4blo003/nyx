# ==========================================================================================
# Author: Pablo González García.
# Created: 18/12/2025
# Last edited: 18/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from typing import List
# External:
import numpy as np
# Internal:
from core.interfaces.vector_store import IVectorStore, IVectorStoreController
from core.interfaces.worker import IBackgroundWorker
from dto.models.vector_store import VectorDocument
from dto.models.retrieval import RetrievalResult
from core.workers.batch import BatchWorker
from core.workers.timer import Timer


# ==============================
# CLASSES
# ==============================

class QdrantVectorStoreController(IVectorStoreController):
    """
    Controller for managing batch insert/delete/update/get  VectorDocuments into a Qdrant
    vector store.
    """
    # ---- Default ---- #

    def __init__(
        self,
        vector_store:IVectorStore,
        batch_size:int = 100,
        max_delay:float = 1
    ) -> None:
        """
        Initialize the controller.

        Args:
            vector_store (IVectorStore): The vector store instance to insert documents into.
            batch_size (int): Maximum number of documents per batch.
            max_delay (float): Maximum delay in seconds before a batch is flushed.
        """
        # Initialize the properties.
        self._vector_store = vector_store
        self._batch_size = batch_size
        self._max_delay = max_delay

        self._queue:asyncio.Queue[VectorDocument] = asyncio.Queue(maxsize=self._batch_size * 10)

        self._flush:asyncio.Event = asyncio.Event()

        self._is_running:bool = False
        self._timer:IBackgroundWorker|None = None
        self._worker:IBackgroundWorker|None = None

    # ---- Methods ---- #

    async def initialize(self) -> None:
        """
        Initialize and starts the background worker for processing batches.
        """
        # Sets it's running.
        self._is_running = True
        # Initializes the background workers.
        self._timer = Timer(
            notify_event=self._flush,
            delay=self._max_delay
        )
        self._worker = BatchWorker(
            queue=self._queue,
            notify_event=self._flush,
            batch_size=self._batch_size,
            callback=self._vector_store.flush
        )

        # Starts workers in background tasks.
        await self._timer.start()
        await self._worker.start()

    async def add_document(
        self,
        doc: VectorDocument
     ) -> None:
        """
        Adds a single document to the queue. for batch processing.

        Args:
            doc (VectorDocument): The document to add.
        """
        # Add the document to the queue.
        self._queue.put_nowait(doc)
        # Checks if the queue is full.
        if self._queue.qsize() >= self._batch_size: self._flush.set()

    async def add_documents(
        self,
        docs: List[VectorDocument]
    ) -> None:
        """
        Adds multiple documents to the queue for batch processing.

        Args:
            docs (List[VectorDocument]): The list of documents to add.
        """
        
        # Iterates over the documents.
        for doc in docs:
            # Add the document to the queue.
            self._queue.put_nowait(doc)
        
        # Checks if the queue is full.
        if self._queue.qsize() >= self._batch_size: self._flush.set()

    async def search(
        self,
        query_embedding:np.typing.NDArray[np.float32],
        limit:int
    ) -> List[RetrievalResult]:
        """
        """
        # Make the search call to the vector store.
        return await self._vector_store.search(
            query_embedding=query_embedding,
            limit=limit
        )

    async def cleanup(self) -> None:
        """
        Stop the worker and flushes any remaining documents in the queue.
        """
        # Sets it's not running.
        self._is_running = False
        # Trigger the event to unblock the worker if waiting.
        self._flush.set()

        # Stops the background workers.
        if self._worker: await self._worker.stop()
        if self._timer: await self._timer.stop()