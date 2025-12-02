import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    """
    Genera muestras de predicción para tareas de pronóstico.
    
    Args:
        features (np.ndarray): Array de características de entrada.
        data (np.ndarray): El array de datos brutos.
        pred_len (int): La longitud del horizonte de predicción.
        drop (int, optional): Número de muestras iniciales a descartar. Por defecto es 0.
        
    Returns:
        tuple: Una tupla que contiene:
            - features (np.ndarray): Muestras de características reestructuradas.
            - labels (np.ndarray): Muestras de etiquetas reestructuradas correspondientes al horizonte de predicción.
    """
    # Obtener el número de columnas (características) en los datos
    n = data.shape[1]
    # Recortar características para excluir los últimos 'pred_len' pasos de tiempo ya que no se pueden usar para predicción
    features = features[:, :-pred_len]
    # Crear etiquetas apilando versiones desplazadas de los datos para cada paso en el horizonte de predicción
    # Esto crea una ventana deslizante de objetivos
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    # Descartar las 'drop' muestras iniciales de las características para manejar relleno o periodos de calentamiento
    features = features[:, drop:]
    # Descartar las 'drop' muestras iniciales de las etiquetas para alinear con las características
    labels = labels[:, drop:]
    # Reestructurar características a array 2D (muestras, dim_característica) y etiquetas a array 2D (muestras, horizonte_aplanado)
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    """
    Calcula métricas de evaluación (MSE y MAE) entre predicciones y objetivos.
    
    Args:
        pred (np.ndarray): Valores predichos.
        target (np.ndarray): Valores reales (ground truth).
        
    Returns:
        dict: Un diccionario que contiene las puntuaciones 'MSE' y 'MAE'.
    """
    return {
        # Calcular Error Cuadrático Medio: media de las diferencias al cuadrado
        'MSE': ((pred - target) ** 2).mean(),
        # Calcular Error Absoluto Medio: media de las diferencias absolutas
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    """
    Evalúa el modelo de pronóstico en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        model (object): La instancia del modelo TS2Vec o compatible con un método 'encode'.
        data (np.ndarray): El conjunto de datos completo.
        train_slice (slice): Objeto slice para índices de datos de entrenamiento.
        valid_slice (slice): Objeto slice para índices de datos de validación.
        test_slice (slice): Objeto slice para índices de datos de prueba.
        scaler (object): Objeto escalador usado para normalización de datos (debe tener inverse_transform).
        pred_lens (list): Lista de longitudes de predicción (horizontes) a evaluar.
        n_covariate_cols (int): Número de columnas covariables a excluir de los objetivos.
        
    Returns:
        tuple: Una tupla que contiene:
            - out_log (dict): Diccionario con predicciones y valores reales, tanto brutos como normalizados.
            - eval_res (dict): Diccionario con métricas de evaluación e información de tiempos.
    """
    # Definir tamaño de relleno para la ventana deslizante
    padding = 200
    
    # Registrar tiempo de inicio para inferencia
    t = time.time()
    # Generar representaciones usando el método encode del modelo
    all_repr = model.encode(
        data,
        causal=True, # Usar codificación causal (sin fugas del futuro)
        sliding_length=1, # Tamaño del paso para ventana deslizante
        sliding_padding=padding, # Relleno para la ventana
        batch_size=256 # Tamaño del lote para codificación
    )
    # Calcular tiempo de inferencia para el modelo TS2Vec
    ts2vec_infer_time = time.time() - t
    
    # Dividir representaciones en conjuntos de entrenamiento, validación y prueba basados en slices
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    # Dividir datos brutos en conjuntos de entrenamiento, validación y prueba, excluyendo columnas covariables
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    # Inicializar diccionarios para almacenar resultados y tiempos
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    # Iterar sobre cada longitud de horizonte de predicción
    for pred_len in pred_lens:
        # Generar muestras de entrenamiento (características y etiquetas) con relleno descartado
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        # Generar muestras de validación
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        # Generar muestras de prueba
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        # Registrar tiempo de inicio para entrenamiento de regresión lineal
        t = time.time()
        # Ajustar un modelo de regresión Ridge usando el protocolo de evaluación
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        # Calcular tiempo de entrenamiento para la regresión lineal
        lr_train_time[pred_len] = time.time() - t
        
        # Registrar tiempo de inicio para inferencia de regresión lineal (predicción)
        t = time.time()
        # Predecir sobre las características de prueba
        test_pred = lr.predict(test_features)
        # Calcular tiempo de inferencia para la regresión lineal
        lr_infer_time[pred_len] = time.time() - t

        # Definir la forma original para reestructurar predicciones: (num_series, -1, pred_len, num_targets)
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        # Reestructurar predicciones a la forma original
        test_pred = test_pred.reshape(ori_shape)
        # Reestructurar etiquetas reales a la forma original
        test_labels = test_labels.reshape(ori_shape)
        
        # Transformación inversa de predicciones y etiquetas a escala original
        if test_data.shape[0] > 1:
            # Si hay múltiples series temporales, intercambiar ejes para coincidir con el formato de entrada esperado del escalador
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            # Si es una sola serie temporal, transformación inversa directa
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)
            
        # Almacenar predicciones y valores reales (ambos normalizados y brutos) en el registro
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        # Calcular y almacenar métricas para datos tanto normalizados como brutos
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    # Compilar todos los resultados de evaluación en un diccionario
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    # Devolver el registro detallado y los resultados de evaluación
    return out_log, eval_res