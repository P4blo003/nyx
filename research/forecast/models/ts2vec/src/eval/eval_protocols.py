# ==========================================================================================
# Author: Pablo González García.
# Created: 02/12/2025
# Last edited: 02/12/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List
# Externos:
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# ==============================
# FUNCIONES
# ==============================

def fit_ridge(
    train_features:np.ndarray,
    train_y:np.ndarray,
    valid_features:np.ndarray,
    valid_y:np.ndarray,
    max_samples:int=10000
) -> Ridge:
    """
    Entrena un modelo de regresión Ridge eligiendo el mejor valor alpha en base a un
    conjunto de validación.

    Args:
        train_features (numpy.ndarray): Características de entrenamiento (N1, D).
        train_y (numpy.ndarray): Etiquetas de entrenamiento (N1, ).
        valid_features (numpy.ndarray): Características de validación (N2, D).
        valid_y (numpy.ndarray): Etiquetas de validación (N2, ).
        max_samples (int): Máximo número de muestras a usar para el entrenamiento.
    
    Returns:
        sklearn.linear_model.Ridge: Modelo de ridge entrenado con el mejor alpha.
    """
    # Submuestrea el conjunto de entrenamiento si es necesario.
    if train_features.shape[0] > max_samples:
        split = train_test_split(
            train_features,
            train_y,
            train_size=max_samples,
            random_state=0
        )
        valid_features, valid_y = split[0], split[2]
    
    # Valores posibles de alpha a evaluar.
    alphas:List[float] = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # Variable para almacenar resultados.
    valid_results:List = []

    # Obtiene el rango máximo.
    range_y = valid_y.max() - valid_y.min()

    # Seevalúa cada alpha entrenando en train y validando en valid.
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)

        # Combina RMSE y MAE.
        rmse:float = np.sqrt(((valid_pred - valid_y) ** 2).mean()) / range_y
        mae:float = np.abs(valid_pred - valid_y).mean() / range_y
        mape = np.mean(np.abs((valid_pred - valid_y) / valid_y)) * 100
        # Calcula la puntuación y la añade.
        score = rmse + mae
        valid_results.append(score)

        # Imprime información.
        print(f"Alpha |{alpha}:\tMape={mape:.2f}%\tScore={score:.2f}")

    # Se elige el alpha que dio mejor rendimiento.
    best_alpha:float = alphas[np.argmin(valid_results)]

    # Reentrena el modelo final con el alpha óptimo.
    lr:Ridge = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr
        