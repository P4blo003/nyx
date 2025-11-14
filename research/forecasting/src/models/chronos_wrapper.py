# ==========================================================================================
# Creation: 14/11/2025
# Last edit: 14/11/2025
# Author: Pablo González García.
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List, Optional
from enum import Enum
# External:
from pandas import DataFrame, Series
from chronos import Chronos2Pipeline
# Internal:
from .base_model import BaseForecastingModel


# ==============================
# ENUMS
# ==============================

class ChronosModel(Enum):
    """
    Supported models.
    """
    # ---- Attributes ---- #
    CHRONOS_2           = "amazon/chronos-2"
    CHRONOS_BOLT_TINY   = "amazon/chronos-bolt-tiny"
    CHRONOS_BOLT_MINI   = "amazon/chronos-bolt-mini"
    CHRONOS_BOLT_SMALL  = "amazon/chronos-bolt-small"
    CHRONOS_BOLT_BASE   = "amazon/chronos-bolt-base"
    CHRONOS_T5_TINY     = "amazon/chronos-t5-tiny"
    CHRONOS_T5_MINI     = "amazon/chronos-t5-mini"
    CHRONOS_T5_SMALL    = "amazon/chronos-t5-small"
    CHRONOS_T5_BASE     = "amazon/chronos-t5-base"
    CHRONOS_T5_LARGE    = "amazon/chronos-t5-large"

class ChronosDevice(Enum):
    """
    Device to run inference on.
    """
    # ---- Attributes ---- #
    CUDA        = "cuda"
    CPU         = "cpu"


# ==============================
# CLASSES
# ==============================

class ChronosWrapper(BaseForecastingModel):
    """
    Encapsulates Chronos2Pipeline models. Provides a consistent interface
    compatible with BaseForecastingModel.
    """
    # ---- Default ---- #

    def __init__(self,
        model:ChronosModel = ChronosModel.CHRONOS_2,
        device:ChronosDevice = ChronosDevice.CPU,
        quantile_levels:List[float] = [0.1, 0.5, 0.9]
    ) -> None:
        """
        Initializes class properties.

        Args:
            model (str): Chronos model.
            device (str): Device to run inference on.
            quantile_levels (List[float]): Quantiles for probabilistic forecast.
        Raises:
            ValueError: The given paramethers are incorrect.
        """        
        # Initialies class properties.
        super().__init__(name=model.value)
        self.__device:str = device.value                    # Device to run inference on.
        self.__quantile_levels:List = quantile_levels       # Quantiles for probabilistic forecast.
        self.__pipeline:Optional[Chronos2Pipeline] = None   # Chronos pipeline.
        self.__isTrained:bool = False                       # If the model is trained.
    

    # ---- Methods ---- #

    def fit(self, x:DataFrame, y:Series) -> None:
        """
        Train the Chronos model on the given time series data.

        Args:
            x (DataFrame): Data to train the model.
            y (Series): Not used. Included for interface compatibility.
        """
        # Copy the dataframe.
        self.__context:DataFrame = x.copy()

        # Checks if the dataframe does not contains 'target' column.
        if 'target' not in self.__context.columns:
            # Sets the target column.
            self.__context['target'] = y.values
        
        # Load pretrained models.
        self.__pipeline = Chronos2Pipeline.from_pretrained(self.Name, device_map=self.__device)
        self.__isTrained = True
        
    def predict(self, 
        horizon: int,
        futures:Optional[DataFrame] = None,
        id_column:str = "id",
        timestamp_column:str = "Timestamp",
        target_column:str = "Target"
    ) -> DataFrame:
        """
        Forecast using Chronos model.

        Args:
            horizon (int): Number of steps to predict.
            futures (Optional[DateFrame]): Future covariates (without target).
            id_column (str): Column identifying different series.
            timestamp_column (str): Column with datetime information.
            target_column (str): Column containing the target values.
        Raises:
            RuntimeError: In case the model is not trained.
        Returns:
            DataFrame: Predicted dataframe.
        """
        # Check if the model is trained.
        if not self.__isTrained or not self.__pipeline:
            # Raise an error.
            raise RuntimeError("Model no trained. Call fit() first.")

        # Make the prediction.
        predicted_df:DataFrame = self.__pipeline.predict_df(
            self.__context,
            future_df=futures,
            prediction_length = horizon,
            quantile_levels = self.__quantile_levels,
            id_column=id_column,
            timestamp_column = timestamp_column,
            target = target_column
        )

        # Returns the predicted dataframe.
        return predicted_df