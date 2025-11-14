# ==========================================================================================
# Creation: 14/11/2025
# Last edit: 14/11/2025
# Author: Pablo González García.
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================


# ==============================
# CLASSES
# ==============================

class ForecastingExperiment:
    """
    Main class to execute forecasting experiments.
    """
    # ---- Default ---- #

    def __init__(self,
        data_set:str,
        model_name:str
    ) -> None:
        """
        Initializes class properties.
        """
        # Initializes class properties.
        self.__data_set:str = data_set
        self.__model_name:str = model_name
    

    # ---- Methods ---- #
    
    def Run(self) -> None:
        """
        Execute the experiment's complete pipeline.
        """
        pass