# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict, Any
# External:
from transformers import ZeroShotClassificationPipeline
from transformers import pipeline
# Internal:
from typing import List
from core.interfaces.classifier import IClassifier


# ==============================
# CLASSES
# ==============================

class ModelClassifier(IClassifier):
    """
    
    """
    # ---- Default ---- #

    def __init__(
        self,
        model_path:str
    ) -> None:
        """
        Initializes the classifier.

        Args:
            model_uri (str): Where the model is downloaded.
        """
        # Initializes the properties.
        self._pipeline:ZeroShotClassificationPipeline = pipeline(
            "zero-shot-classification",
            model=model_path,
            device=0
        )


    # ---- Methods ---- #

    def classify(self, query: str, threshold: float = 0.7) -> List[str]:
        """
        Classify the query into multiclass values.

        Args:
            query (str): Query to classify.
            threshold (float): Threshold.
        """
        # Classify the query.
        results:Dict[str,Any] = self._pipeline(query)

        # Gets labels with score gratter than threshold.
        labels:List[str] = [
            label for label, score in zip(results['labels'], results['scores'])     # type:ignore
            if score >= threshold
        ]

        return labels