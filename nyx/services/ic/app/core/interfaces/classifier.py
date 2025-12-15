# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
from abc import ABC
from abc import abstractmethod


# ==============================
# INTERFACES
# ==============================

class IClassifier(ABC):
    """
    """
    # ---- Methods ---- #

    @abstractmethod
    def classify(self, query:str, threshold:float=0.7) -> List[str]:
        """
        
        Args:
            query (str): Query to classify.
            threshold (float): Threshold.

        Returns:
            List[str]: List of query's classes.
        """
        pass