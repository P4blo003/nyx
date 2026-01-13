# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path
from typing import List
# External:
from transformers import AutoTokenizer


# ==============================
# CLASSES
# ==============================

class TokenizerService:
    """
    """

    # ---- Constructor ---- #

    def __init__(
        self,
        model_dir:str|Path
    ) -> None:
        """
        Initialize the instance.

        Args:
            model_dir (str|Path): 
        """

        # Initializes properties.
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
    

    # ---- Methods ---- #

    def tokenize(
        self,
        queries:str|List[str]
    ):
        """"""

        # Convert que single query into a list of one string.
        if isinstance(queries, str): queries = [queries]

        # Tokenize the queries.
        batch_encoding = self._tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        # Returns values.
        return batch_encoding["input_ids"]