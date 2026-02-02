# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
import tempfile
from typing import BinaryIO, List

# External:
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Internal:
from domain.ports.document import IDocumentProcessor
from domain.entities.document import ProcessedChunk


# ==============================
# CLASSES
# ==============================

class UnstructuredProcessor(IDocumentProcessor):
    """
    
    """

    # ---- Methods ---- #

    def process(self, file: BinaryIO, filename: str) -> List[ProcessedChunk]:
        """
        
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            elements = partition(filename=tmp_path)

            chunks = chunk_by_title(elements=elements)

            results:List[ProcessedChunk] = []
            for chunk in chunks:
                results.append(ProcessedChunk(
                    text=str(chunk),
                    metadata=chunk.metadata.to_dict()
                ))

            return results

        finally: os.remove(tmp_path)