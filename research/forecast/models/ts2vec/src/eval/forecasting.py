# ==========================================================================================
# Author: Pablo González García.
# Created: 04/12/2025
# Last edited: 04/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import Dict, Tuple
# Internos:
import numpy as np


# ==============================
# FUNCIONES
# ==============================

def generate_prediction_samples(
    features:np.ndarray,
    data:np.ndarray,
    prediction_length:int,
    drop:int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara los datos y embeddings para el entrenamiento de una capa de predicción
    de series temporales a partir de las representaciones generadas por el encoder.

    Esta función alinea cada embedding temporal (feature) con la secuencia de
    'prediction_length` pasos futuros de la serie temporal original, y luego
    aplana las dimensiones de batch y tiempo.

    Args:
        features (np.ndarray): Embeddings temporales generados por el encoder,
            con la forma de (batch_size, timesteps, hidden_dims).
        data (np.ndarray): Serie temporal original.
        prediction_length (int): Longitud de la serie temporal que se desa
            predecir a partir de cada embedding (número de pasos futuros).
        drop (int, optional): Número de timesteps iniciales a descartar de ambas
            secuencias (features y labels). Útil para descartar datos con padding
                o inestables.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Una tupla con las features y los labels
            alineados y aplanados.
            - features_out: (n_samples, hidden_dim)
            - labels_out: (n_samples, prediction_length * features_dim)
    """
    # Obtiene el número de timesteps.
    n:int = data.shape[1]

    # Elimina los últimos `prediciont_length` de los embeddings.
    features = features[:, :, :-prediction_length]

    # Crea una matriz de etiquetas donde cada fila contiene `prediction_length` pasos.
    # Se elimina el primero [:, 1:] ya que el primer timestep no tiene embedding
    # anterior.
    labels:np.ndarray = np.stack(
        arrays=[data[:, i:1 + n + i - prediction_length] for i in range(prediction_length)],
        axis=2
    )[:, 1:]

    # Descarta los primeros 'drop' timesteps de las features y labels.
    features = features[:, drop:]
    labels = labels[:, drop:]

    # Aplana las dimensiones de batch y timesteps en una sola dimensión
    # creando un conjunto de muestras listas para la regresión supervisada.
    features_out:np.ndarray = features.reshape(-1, features.shape[-1])
    labels_out:np.ndarray = labels.reshape(-1, labels.shape[2] * labels.shape[3])

    return features_out, labels_out

def calculate_metrics(
    prediction:np.ndarray,
    target:np.ndarray
) -> Dict[str, float]:
    """
    Calcula el error cuadrático medio `MSE` y el error Absoluto medio `MAE` entre
    las predicciones y los valores objetivo.

    Esta función opera sobre arrays de NumPy y se utiliza típicamente para evaluar
    el rendimiento de modelos de regresión.

    Args:
        prediction (np.ndarray): Array de numpy que contiene las predicciones del modelo,
            con la forma de (timesteps, featres) o aplanado.
        target (np.ndarray): Array de numpy que contiene los valores reales u objetivo.
            Debe tener la misma forma que `prediction`.
        
    Returns:
        Dict[str, any]: Diccionario con las métricas calculadas:
            - `MSE`: Mean Squared Error (Error cuadrático medio).
            - `MAE`: Mean Absolute Error (Error absoluto medio).
    """
    # Se calcula el MSE y MAE.
    mse:float = ((prediction - target) ** 2).mean()
    mae:float = np.abs(prediction - target).mean()

    # Genera el diccionario.
    return {
        'MSE':mse,
        'MAE':mae
    }