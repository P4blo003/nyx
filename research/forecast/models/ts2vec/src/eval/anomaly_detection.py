# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import time
from typing import List, Dict
# Externos:
import numpy as np
import bottleneck as bn
from sklearn.metrics import f1_score, precision_score, recall_score
# Internos:
from ts2vec import TS2Vec
from mask import MaskMode


# ==============================
# FUNCIONES
# ==============================

def get_range_proba(
    predict:np.ndarray,
    label:np.ndarray,
    delay:int=7
) -> np.ndarray:
    """
    Ajusta las predicciones de anomalías considerando un delay permitido.

    Args:
        predict (): Array con predicciones curdas del modelo.
        label (np.ndarray): Array con etiquetas reales, indicando intervalos de anomalía.
        delay (int): Número máximo de pasos de tolerancia para aceptar que el modelo
            detectó un intervalo.
    
    Returns:
        numpy.ndarray: Array donde cada intervalo de anomalía está marcado como 1 si se detecto al menos una
            anomalía dentro del rango permitido. Si no se marca 0.
    """
    # Detecta los índices donde la etiqueta cambia,
    splits = np.where(label[1:] != label[:-1])[0] + 1
    # Determina si la secuencia empieza dentro de un segmento de anomalía.
    is_anomaly:bool = label[0] == 0

    # Hace una copua de las predicciones originales para modificarlas sin alterar el input.
    new_predict:np.ndarray = np.array(predict)

    # Almacena la posición inicial del segmento actual.
    pos:int = 0

    # Itera sobre cada frontera entre segmentos homogéneos.
    for sp in splits:
        # Solo se aplica si estamos en un intervalo de anomalía.
        if is_anomaly:
            # Comprueba si dentro del tramo inicial del segmento aparece una predicción
            # positiva.
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                # Se marca todo el intervalo real como detectado.
                new_predict[pos:sp] = 1
            # Si el modelo no detecto nada dentro del rango.
            else:
                # Se marca como no detectado.
                new_predict[pos: sp] = 0
        
        # Cambia el estado del segmento.
        is_anomaly = not is_anomaly
        # Actualiza el inicio del siguiente segmento.
        pos = sp
    
    # Define el final de la secuencia para tratar el último segmento.
    sp = len(label)

    # Si la secuencia termina en un intervalo de anomalía.
    if is_anomaly:
        # Comprueba si dentro del tramo inicial del segmento aparece una predicción
        # positiva.
        if 1 in predict[pos:min(pos + delay + 1, sp)]:
            # Se marca todo el intervalo real como detectado.
            new_predict[pos:sp] = 1
        # Si el modelo no detecto nada dentro del rango.
        else:
            # Se marca como no detectado.
            new_predict[pos: sp] = 0
    
    # Devuelve las predicciones.
    return new_predict

def reconstruct_label(
    timestamp:np.ndarray,
    label:np.ndarray 
) -> np.ndarray:
    """
    Reconstruye una secuencia de etiquetas (labels) en un eje temporal uniforme, a partir
    de timestamps que pueden no estar regularmente espaciados o pueden estar desordenados.

    Args:
        timestamp (numpy.ndarray): Array con los timestamps originales.
        label (np.ndarray ): Array con las etiquetas correspondientes.
    
    Returns:
        numpy.ndarray: Array de tamaño fijo que reconstruye la secuencia temporal completa.
    """
    # Convierte los timestamps en un array de numpy
    timestamp = np.asarray(timestamp, np.int64)
    # Obtiene los índices que ordenaría el array de timestamps.
    index:np.ndarray = np.argsort(timestamp)

    # Aplica orden y obtiene un array de timestamps ordenados ascendentemente.
    timestamp_sorted:np.ndarray = np.asarray(timestamp[index])
    # Calcula el mínimo intervalo entre timestamps consecutivos.
    interval = np.min(np.diff(timestamp_sorted))

    # Convierte las etiquetas en array de numpy.
    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    # Convierte cada timestamp en un índice uniforme.
    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    # Crea un array de 0 que representará todas la línea temporal.
    new_label:np.ndarray = np.zeros(
        shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,),
        dtype=np.int64
    )
    # Asigna las etiquetas reales a las posiciones que corresponden a sus índices temporales.
    new_label[idx] = label

    # Retorna el valor.
    return new_label

def eval_add_result(
    test_pred_list:List[np.ndarray],
    test_labels_list:List[np.ndarray],
    test_timestamps_list:List[np.ndarray],
    delay:int
) -> Dict[str, float]:
    """
    Evalúa los resultados de un modelo de detección de anomalías en series temporales,
    combinando múltiples subsecuencias (batches, ventanas).
    """
    # Variables para acumular las etiquetas.
    labels = []
    pred = []

    # Itera simultánetamente sobre los tres conjuntos.
    for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
        # Comprueba que tengan la misma forma.
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        # Reconstruye la secuencia de etiquetas usando los timestamps.
        test_labels = reconstruct_label(
            timestamp=test_timestamps,
            label=test_labels
        )
        # Reconstruye la secuencia de etiquetas usando los timestamps.
        test_pred = reconstruct_label(
            timestamp=test_timestamps,
            label=test_pred
        )
        # Corrige o ajusta las predicciones.
        test_pred = get_range_proba(
            predict=test_pred,
            label=test_labels,
            delay=delay
        )

        # Acumula las predicciones y etiquetas reconstruidas.
        labels.append(test_labels)
        pred.append(test_pred)

    # Concatena todas las ventanas/batches en dos arrays.
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)

    # Retorna el diccionario.
    return {
        'f1': float(f1_score(
            y_true=labels,
            y_pred=pred
        )),
        'precission': float(precision_score(
            y_true=labels,
            y_pred=pred
        )),
        'recall': float(recall_score(
            y_true=labels,
            y_pred=pred
        ))
    }

def np_shift(
    arr:np.ndarray,
    num:int,
    fill_value:float = np.nan
) -> np.ndarray:
    """
    Desplaza los elementos de un array de numpy hacia la derecha o izquierda, rellenando
    los huevos con un valor especificado.

    Args:
        arr (numpy.ndarray): Array de numpy que hay que desplazar.
        num (int): Si > 0 se desplaza a la derecha `num` unidades. Si < 0 se
            desplaza a la izquierda `num` unidades.
        fill_value (int): Valor con el que rellenar.
    
    Returns:
        numpy.ndarray: Array de numpy desplazado.
    """
    # Crea un array vacío del mismo valor.
    result:np.ndarray = np.empty_like(arr)

    # Desplazamiento a la derecha.
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    # Desplazamiento a la izquierda.
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    # Si no hay desplazamiento.
    else: result[:] = arr

    # Retorna el resultado.
    return result

def eval_anomaly_detection(
    model:TS2Vec,
    all_train_data,
    all_train_labels,
    all_train_timestamps,
    all_test_data,
    all_test_labels,
    all_test_timestamps,
    delay:int
):
    """
    Ejecuta un proceso completo de detección de anomalías usando el modelo
    TS2Vec. Combina datos de entrenamiento y prueba, genera embeddings con
    y sin máscara, calcula el error reconstruido y deriva una señal binaria
    de anomalía.

    Args:
        model (TS2Vec): Modelo TS2Vec que realiza los cálculos.
        all_train_data ():
        all_train_labels ():
        all_train_timestamps ():
        all_test_data ():
        all_test_labels ():
        all_test_timestamps ():
        delay (int):
    """
    # Varriables para almacenar las representaciones.
    all_train_repr:Dict = {}
    all_test_repr:Dict = {}
    all_train_repr_wom:Dict = {}        # Sin máscara.
    all_test_repr_wom:Dict = {}         # Sin máscara.
    
    # Obtiene el tiempo inicial.
    start:float = time.time()

    # Recorre todos los datos de entrenamiento.
    for k in all_train_data:
        # Obtiene la serie de entrenamiento y prueba asociadas.
        train_data = all_train_data[k]
        test_data = all_test_data[k]

        # Calcula los embeddings (Con máscara).
        full_repr = model.encode(
            data=np.concatenate([train_data, test_data]).reshape(1, -1, 1),
            mask=MaskMode.MASK_LAST,
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        # Separa en entrenamiento y test.
        all_train_repr[k] = full_repr[:len(train_data)]
        all_test_repr[k] = full_repr[len(train_data):]

        # Calcula los embeddings (Sin máscara).
        full_repr_wom = model.encode(
            data=np.concatenate([train_data, test_data]).reshape(1, -1, 1),
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        # Separa en entrenamiento y test.
        all_train_repr_wom[k] = full_repr_wom[:len(train_data)]
        all_test_repr_wom[k] = full_repr_wom[len(train_data):]

    # Listas donde se almacenan resultados, etiquetas y timestamps.
    res_log:List[np.ndarray] = []
    labels_log:List[np.ndarray] = []
    timestamps_log:List[np.ndarray] = []

    # Itera para cada serie temporal.
    for k in all_train_data:
        # Carga datos y etiquetas.
        train_data = all_train_data[k]
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        # Calcula el error reconstruido.
        train_err = np.abs(all_train_repr_wom[k] - all_train_repr[k]).sum(axis=1)
        test_err = np.abs(all_test_repr_wom[k] - all_test_repr[k]).sum(axis=1)

        # Media móvil con ventana 21.
        ma = np_shift(
            arr=bn.move_mean(
                np.concatenate([train_err, test_err]),
                21
            ),
            num=1
        )
        # Normaliza el error relativo respecto a la media local.
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        # Umbral estadístico.
        threshold:float = np.mean(train_err_adj) + 4 * np.std(train_err_adj)

        # Señal binaria, devuelve 1 si hay anomalía.
        test_res = (test_err_adj > threshold) * 1

        # Si en los últimos delay pasos hubo anomalía, se fuerza ventana de cooldown.
        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1: test_res[i] = 0
        
        # Se almacenan los resultados, las etiquetas reales y los timestmaps.
        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
    
    # Obtiene la duración.
    duration:float = time.time() - start

    # Calculo de métricas de evaluación.
    eval_res:Dict[str, float] = eval_add_result(
        test_pred_list=res_log,
        test_labels_list=labels_log,
        test_timestamps_list=timestamps_log,
        delay=delay
    )
    eval_res['infer_time'] = duration

    # Retorna la señal de anomalía y las métricas.
    return res_log, eval_res