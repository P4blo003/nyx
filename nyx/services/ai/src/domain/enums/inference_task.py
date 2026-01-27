# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from enum import StrEnum


# ==============================
# ENUMS
# ==============================

class InferenceTask(StrEnum):
    """
    Enumeration of possible inference tasks supported by an engine or model.

    Attributes:
        EMBEDDING (str): Task where the model generates vector embeddings.
        CLASSIFICATION (str): Task where the model predicts categories or labels
            from input data.
        TEXT_GENERATION (str): Task where the model generates text from input
            prompts.
    """

    # ---- Attributes ---- #

    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    TEXT_GENERATION = "text_generation"