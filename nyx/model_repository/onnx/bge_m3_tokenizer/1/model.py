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
MODEL_NAME:str = "bge_m3_model"


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
            trust_remote_code=True
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

            # Ensures 2D.
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Only squeeze if output is 3D.
            if input_ids.ndim == 3:
                input_ids = np.squeeze(input_ids, axis=1)
                attention_mask = np.squeeze(attention_mask, axis=1)
            
            input_ids_tensor = pb_utils.Tensor("input_ids", input_ids.astype(np.int64))
            attention_mask_tensor = pb_utils.Tensor("attention_mask", attention_mask.astype(np.int64))

            # Generates response.
            response = pb_utils.InferenceResponse(output_tensors=[input_ids_tensor, attention_mask_tensor])
            responses.append(response)

        return responses