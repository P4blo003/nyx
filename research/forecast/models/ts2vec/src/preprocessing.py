# ==========================================================================================
# Author: Pablo González García.
# Created: 25/11/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List
# Externos:
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==============================
# FUNCIONES
# ==============================

def get_time_features(
    df:pl.DataFrame,
    timestamp_column:str
) -> np.ndarray:
    """
    Extrae características de tiempo de la columna timestamp y devuelve un array
    de numpy con ellas.

    Args:
        df (polars.DataFrame): DataFrame de polars.
        timestamp_column (str): Nombre de la columna temporal.
    """
    # Crea las características en función del timestamp.
    return df.select([
        pl.col(timestamp_column).dt.second().alias("second"),
        pl.col(timestamp_column).dt.minute().alias("minute"),
        pl.col(timestamp_column).dt.hour().alias("hour"),
        pl.col(timestamp_column).dt.weekday().alias("weekday"),
        pl.col(timestamp_column).dt.day().alias("day"),
        pl.col(timestamp_column).dt.ordinal_day().alias("dayofyear"),
        pl.col(timestamp_column).dt.month().alias("month"),
        pl.col(timestamp_column).dt.week().alias("weekofyear")
    ]).to_numpy().astype(np.float32)



# ==============================
# CLASES
# ==============================

class Preprocessor:
    """
    Clase encargada de preprocesar los datasets para convertirlos en un
    array de numpy válidos para TS2Vec.
    """
    # ---- Default ---- #

    def __call__(
        self,
        df:pl.DataFrame,
        train_size:float,
        timestamp_column:str = "Timestamp",
        timestamp_format:str = "%Y-%m-%d %H:%M:%S",
        categorical_columns:List[str]|None = None
    ):
        """
        Peprocesa el DataFrame y lo convierte en un array de numpy válido para
        TS2Vec.

        Args:
            df (polars.DataFrame): DataFrame a procesar.
            train_size (float): Porcentaje del dataset empleado para el entrenamiento.
            timestamp_column (str): Nombre de la columna temporal.
            timestamp_format (str): Formato de la fecha de `timestamp_column`.
            categorical_columns (List[str]|None) = Lista con las columnas categóricas.
        """
        # Ordena a partir de la columna.
        df = df.sort(by=timestamp_column).with_columns(pl.col(timestamp_column).str.to_datetime(timestamp_format))
        
        # Genera características temporales en función del timestamp.
        ts_features:np.ndarray = get_time_features(
            df=df,
            timestamp_column=timestamp_column
        )
        num_ts_features:int = ts_features.shape[-1]

        # One-Hot de variables categóricas.
        if categorical_columns is not None:
            # Recorre las columnas categóricas.
            for col in categorical_columns:
                # Selecciona timestamp + columna categórica
                tmp = df.select([timestamp_column, col])
                # Aplica One-Hot solo para la columna categórica
                dummies = tmp.select(col).to_dummies()
                # Añade el timestamp al DataFrame de dummies.
                dummies = dummies.with_columns(tmp[timestamp_column])
                # Hace join a partir de la columna de timestamp.
                df = df.join(dummies, on=timestamp_column).drop(col)

        # Convierte a numpy (todas las columnas menos la de timestamp).
        final_cols:List[str] = [c for c in df.columns if c != timestamp_column]
        data:np.ndarray = df.select(final_cols).to_numpy()

        # Obtiene el tamaño del array. (Aquí data tiene la forma (n_timestamps, n_features)).
        n:int = data.shape[0]                           # Obtiene el número de timestamps (n_timestamps).
        # Separa en entrenamiento/validación/test
        train_slice:slice = slice(None, int(train_size * n))
        valid_slice:slice = slice(int(train_size * n), int(((1 - train_size) / 2 + train_size) * n))
        test_slice:slice = slice(int(((1 - train_size) / 2 + train_size) * n), None)

        # Genera el scaler empleando datos solo de entrenamiento.
        scaler:StandardScaler = StandardScaler().fit(data[train_slice])
        # Normaliza los datos solo con información del entrenamiento.
        data = scaler.transform(data)

        # Añade una dimensión (que representa el batch).
        data = np.expand_dims(data, 0)  # (1, n_timestamps, n_features)

        # Comprueba si hay características temporales que añadir.
        if num_ts_features > 0:
            # Genera el scaler empleando datos de entrenamiento.
            ts_scaler:StandardScaler = StandardScaler().fit(ts_features[train_slice])
            # Añade una dimensión a las características temporales.
            ts_features_scaled:np.ndarray = np.expand_dims(ts_scaler.transform(ts_features), 0)
            # Normaliza los datos solo con información del entrenamiento.
            data = np.concatenate([np.repeat(ts_features_scaled, data.shape[0], axis=0), data], axis=-1)

        # Horizontes de predicción [24h (1 día), 48h (2 días), 96h (4 días), 288h (12 días), 672h (28 días)].
        pred_lens:List[int] = [24, 48, 96, 288, 672]

        # Retorna los valores obtenidos.
        return data, train_slice, valid_slice, test_slice, scaler, pred_lens, num_ts_features