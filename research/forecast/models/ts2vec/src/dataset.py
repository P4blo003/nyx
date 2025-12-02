# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 02/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
# Externos:
import polars as pl
# Internos:
from preprocessing import Preprocessor


# ==============================
# CLASES
# ==============================

class DatasetLoader(ABC):
    """
    Clase absracta para cargadores de datasets.
    """
    # ---- Default ---- #

    def __init__(
        self,
        url:str|Path
    ) -> None:
        """
        Inicializa el cargador de datos.

        Args:
            file (str|Path): Ruta al fichero a cargar.
        """
        # Inicializa las propiedades.
        self.url:Path = Path(url)
    

    # ---- Métodos ---- #

    @abstractmethod
    def load(self) -> Any:
        """
        Carga los datos desde la url y los procesa.
        """
        pass

class SIECDatasetLoader(DatasetLoader):
    """
    Cargador de datos para el CSV `Steel Industry Energy Consumption`.
    """
    # ---- Atributos ---- #

    # Ruta al fichero.
    DEFAULT_FILE_URL:str = r"data/raw/steel_industry_energy_consumption_dataset.csv"


    # ---- Default ---- #

    def __init__(self) -> None:
        """
        Inicializa el cargador de datos.
        """
        # Constructor de DatasetLoader.
        super().__init__(url=self.DEFAULT_FILE_URL)
    

    # ---- Métodos ---- #

    def load(self):
        """
        Carga los datos desde la url y los procesa.
        """
        # Carga el csv en un DataFrame de polars.
        df:pl.DataFrame = pl.read_csv(
            source=self.url,
            has_header=True,
            separator=',',
            infer_schema_length=1000
        )

        # Genera el preprocesador.
        preprocessor:Preprocessor = Preprocessor()

        # Retorna los datos preprocesados.
        return preprocessor(
            df=df,
            train_size=0.6,
            timestamp_column='Timestamp',
            timestamp_format="%d/%m/%Y %H:%M",
            categorical_columns=['WS', 'DW', 'LT']
        )