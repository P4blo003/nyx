# ==========================================================================================
# Creation: 14/11/2025
# Last edit: 14/11/2025
# Author: Pablo González García.
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Optional
# External:
from pandas import DataFrame


# ==============================
# CLASSES
# ==============================

class BaseForecastingModel(ABC):
    """
    Abstract base class for forecasting models.
    """
    # ---- Default ---- #

    def __init__(self,name:str) -> None:
        """
        Initializes class properties.

        Args:
            name (str): Model's name.
        """
        # Initializes class properties.
        self.__name:str = name


    # ---- Properties ---- #
    
    @property
    def Name(self) -> str:
        """
        Gets the model name.

        Returns:
            str: Model name.
        """
        # Returns the value.
        return self.__name
    

    # ---- Methods ---- #

    @abstractmethod
    def fit(self, x:DataFrame) -> None:
        """
        Train the model with the given data.
        
        Args:
            df (DataFrame): Data to train the model.
        Raises:
            NoImplementedError: In case this function is not implemented.
        """
        # Raise exception.
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def predict(self, 
        horizon: int,
        futures:Optional[DataFrame] = None,
        id_column:str = "id",
        timestamp_column:str = "Timestamp",
        target_column:str = "Target"
    ) -> DataFrame:
        """
        Predict for a given horizont.

        Args:
            horizon (int): Number of steps to predict.
            futures (Optional[DateFrame]): Future covariates (without target).
            id_column (str): Column identifying different series.
            timestamp_column (str): Column with datetime information.
            target_column (str): Column containing the target values.
        Raises:
            NoImplementedError: In case this function is not implemented.
        Returns:
            DataFrame: Predicted dataframe.
        """
        # Raise exception.
        raise NotImplementedError("This method must be implemented.")