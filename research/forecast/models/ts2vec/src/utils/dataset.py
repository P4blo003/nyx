# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from pathlib import Path
from typing import List
# Externos:
import polars as pl
# Internos:
from utils.preprocessing import Preprocessor


# ==============================
# FUNCIONES
# ==============================

def load_csv(
    dataset:str|Path,
    separator:str = ',',
    has_header:bool = False,
    timestamp_column:str = "Timestamp",
    timestamp_format:str = "%Y-%m-%d %H:%M:%S",
    categorical_columns:List[str]|None = None
):
    """
    Carga el fichero CSV y lo preprocesa.

    Args:
        dataset (str|Path): Ruta del fichero a cargar.
        separator (str): Separador empleado en el dataset.
        has_header (bool): True si incluye cabecera y False en caso contrario.
        timestamp_column (str): Nombre de la columna temporal.
        timestamp_format (str): Formato de la columna temporal.
        categorical_columns (List[str]|None): Lista de las columnas categóricas.
    
    Returns:
        Datos preprocesados y separados.
    """
    # Carga el csv en un DataFrame de polars.
    df:pl.DataFrame = pl.read_csv(
        source=dataset,
        has_header=has_header,
        separator=separator,
        infer_schema_length=1000
    )

    # Genera el preprocesador.
    preprocessor:Preprocessor = Preprocessor()

    # Retorna los datos preprocesados.
    return preprocessor(
        df=df,
        train_size=0.6,
        timestamp_column=timestamp_column,
        timestamp_format=timestamp_format,
        categorical_columns=categorical_columns
    )