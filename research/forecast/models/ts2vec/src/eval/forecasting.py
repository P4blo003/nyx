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
import time
import warnings
from typing import Dict, List
# Externos:
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import LinAlgWarning

# Internos:
from model import TS2Vec
import eval.eval_protocols as eval_protocols


# ==============================
# CONFIGURACIÓN
# ==============================

# Ignorar solo los LinAlgWarnings
warnings.simplefilter("ignore", LinAlgWarning)


# ==============================
# FUNCIONES
# ==============================

def generate_prediction_sample(
    features:np.ndarray,
    data:np.ndarray,
    prediction_length:int,
    drop:int = 0
):
    """
    Genera muestras de predicción para tareas de pronóstico.
    
    Args:
        features (numpy.ndarray): Array de características de entrada.
        data (numpy.ndarray): El array de datos brutos.
        prediction_length (int): La longitud del horizonte de predicción.
        drop (int|None): Número de muestas iniciales a descartar.
    
    Returns:
        Tuple: Una tupla que contiene:
            - features (numpy.ndarray): Muestras de características reestructurada.
            - labels (numpy.ndarray): Muestras de etiquetas reestructuradas correspondientes al
                horizonte de predicción.
    """
    # Obtiene el número de columnas (Características de los datos).
    n:int = data.shape[1]
    # Recorta las característica para excluir los últimos prediction_length pasos de tiempo.
    features = features[:, :-prediction_length]
    # Crea etiquetas apilando versiones desplazadas de los datos para cada paso en el horizonte de predicción.
    # Esto crea una ventana deslizante de objetivos.
    labels:np.ndarray = np.stack([ data[:, i:1+n+i-prediction_length] for i in range(prediction_length)], axis=2)[:, 1:]
    # Descarta las 'drop' muestras iniciales de las características para manejar relleno o periodos de calentamiento.
    features = features[:, drop:]
    # Descarta las 'drop' muestras iniciales de las eqiquetas para alinear con las características.
    labels = labels[:, drop:]
    # Reestructura características a array 2D (muestras, dim_características) y etiquetas a array 2D (muestras, horizonte_aplanado).
    return features.reshape(-1, features.shape[-1]), labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(
    prediction: np.ndarray,
    target:np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de evaluación (MSE y MAE) entre predicciones y objetivos.

    Args:
        prediction (numpy.ndarray): Valores predichos.
        target (numpy.ndarray): Valores reales.
    
    Returns:
        Dict: Diccionario que contiene las puntuaciones 'MSE' y 'MAE'.
    """
    # Calcula los valores.
    return {
        'MSE': ((prediction - target)**2).mean(),
        'MAE': np.abs(prediction - target).mean()
    }

def eval_forecasting(
    model:TS2Vec,
    data:np.ndarray,
    train_slice:slice,
    valid_slice:slice,
    test_slice:slice,
    scaler:StandardScaler,
    prediction_lengths:List[int],
    n_covariates_cols:int
):
    """
    Evalúa el modelo de pronóstico en conjuntos de entrenamiento, validación y prueba.

    Args:
        model (TS2Vec): Instancia del modelo TS2Vec.
        data (numpy.ndarray): Conjunto de datos completos.
        train_slice (slice): Objeto slice para índices de datos de entrenamiento.
        valid_slice (slice): Objeto slice para índices de datos de validación.
        test_slice (slice): Objeto slice para índices de datos de prueba.
        scaler (StandardScaler): Objeto escalador usado para la normalización.
        prediction_lengths (List[int]): Lista de longitudes de predicción (horizontes) a evaluar.
        n_covariate_cols (int): Número de columnas covariables a excluir de los objetivos.
    
    Returns:
        Tuple: Una tupla que contiene:
            - out_log (Dict): Diccionario con predicciones y valores reales, tanto brutos como normalizados.
            - eval_res (Dict): Diccionario con métricas de evaluación e información de tiempos.
    """
    # Define el relleno de la ventana deslizante.
    padding:int = 200

    # Obtiene el tiempo inicial.
    start:float = time.time()

    # Genera representaciones usando el método encode del modelo.
    all_repr = model.encode(
        data=data,
        mask_generator=None,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )

    # Calcula la duración.
    infer_duration:float = time.time() - start

    # Divide las presentaciones en conjuntos de entrenamiento, validación y prueba.
    train_repr:np.ndarray = all_repr[:, train_slice]
    valid_repr:np.ndarray = all_repr[:, valid_slice]
    test_repr:np.ndarray = all_repr[:, test_slice]

    # Divide datos brutos en conjuntos de entrenamiento, validación y prueba, excluyendo columnas covariables.
    train_data:np.ndarray = data[:, train_slice, n_covariates_cols:]
    valid_data:np.ndarray = data[:, valid_slice, n_covariates_cols:]
    test_data:np.ndarray = data[:, test_slice, n_covariates_cols:]

    # Variables para almacenar resultados.
    ours_result:Dict = {}
    lr_train_time:Dict = {}
    lr_infer_time:Dict = {}
    out_log:Dict = {}

    # Itera sobre cada longitud de horizonte de predicción.
    for prediction_length in prediction_lengths:
        # Genera muestras de entrenamiento con relleno descartado.
        train_features, train_labels = generate_prediction_sample(
            features=train_repr,
            data=train_data,
            prediction_length=prediction_length,
            drop=padding
        )
        # Genera muestras de validación.
        valid_features, valid_labels = generate_prediction_sample(
                    features=valid_repr,
                    data=valid_data,
                    prediction_length=prediction_length
                )
        # Genera muestras de prueba.
        test_features, test_labels = generate_prediction_sample(
                    features=test_repr,
                    data=test_data,
                    prediction_length=prediction_length
                )
    
        # Obtiene el tiempo inicial.
        start:float = time.time()
        # Ajustsa el modelo de regresión Ridge usando el protocolo de evaluación.
        lr = eval_protocols.fit_ridge(
            train_features=train_features,
            train_y=train_labels,
            valid_features=valid_features,
            valid_y=valid_labels
        )
        # Almacena la duración.
        lr_train_time[prediction_length] = time.time() - start
        
        # Obtiene el tiempo inicial.
        start:float = time.time()
        #Predice sobre las caracteristicas de prueba.
        test_pred:np.ndarray = lr.predict(test_features)
        # Almacena la duración.
        lr_infer_time[prediction_length] = time.time() - start

        # Define la forma original para reestructurar las predicciones.
        ori_shape = test_data.shape[0], -1, prediction_length, test_data.shape[2]
        # Reestructura predcciones a la forma original.
        test_pred = test_pred.reshape(ori_shape)
        # Reestructura etiquetas reales a la forma original.
        test_labels = test_labels.reshape(ori_shape)

        # Transformación inversa de predicciones y etiquetas a escala original.
        if test_data.shape[0] > 1:
            # Si hay múltiples series temporales, intercambia ejes para coincidir con el formato de entrada
            # esperado.
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
             # Aplanar a 2D (n_samples, n_features)
            shape_orig = test_pred.shape
            test_pred_2d = test_pred.reshape(-1, shape_orig[-1])
            test_labels_2d = test_labels.reshape(-1, shape_orig[-1])

            # Si es una sola serie temporal, transformación inversa directa
            test_pred_inv = scaler.inverse_transform(test_pred_2d)
            test_labels_inv = scaler.inverse_transform(test_labels_2d)
        
        # Almacena predicciones y valores realies.
        out_log[prediction_length] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }

        # Calucla y almacena métricas para datos tanto normalizados como brutos.
        ours_result[prediction_length] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
    
    # Establece todos los resultados en un único diccionario.
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': infer_duration,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    # Devuelve el registro detallado y los resultados de evaluación
    return out_log, eval_res