# ==========================================================================================
# Author: Pablo González García.
# Created: 21/01/2025
# Last edited: 21/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from typing import List

# External:
import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


# ==============================
# VARIABLES
# ==============================

MODELS_PATH:str = "/models"
MODEL_NAME:str = "multilingual_e5_base_model"


# ==============================
# CLASSES
# ==============================


class TritonPythonModel:
    """
    """

    # ---- Methods ---- #

    def initialize(self, args):
        """
        """
            
        # Generates complete path.
        tokenizer_path:str = os.path.join(
            MODELS_PATH,
            MODEL_NAME,
            "1"
        )

        # Initialize class tokenizer.
        self._tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            fix_mistral_regex=True
        )

    def execute(self, requests):
        """
        """

        # List to keep responses.
        responses:List = []

        # Iterate over received requests.
        for request in requests:
            
            # Get text from request.
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in input_tensor.as_numpy()]

            # Tokenization.
            enc = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="np"
            )

            # Verifies correct dimension.
            input_ids = enc["input_ids"].squeeze(axis=0) if enc["input_ids"].ndim == 3 else enc["input_ids"]
            attention_mask = enc["attention_mask"].squeeze(axis=0) if enc["attention_mask"].ndim == 3 else enc["attention_mask"]

            # Create output tensors.
            input_ids_tensors = pb_utils.Tensor("input_ids", input_ids.astype(np.int64))
            attention_mask_tensors = pb_utils.Tensor("attention_mask", attention_mask.astype(np.int64))
            
            # Generates response.
            response = pb_utils.InferenceResponse(output_tensors=[input_ids_tensors, attention_mask_tensors])
            responses.append(response)

        return responses