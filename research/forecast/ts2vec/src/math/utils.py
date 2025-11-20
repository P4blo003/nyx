# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
# Last edited: 20/11/2025
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
import torch


# ==============================
# FUNCIONES
# ==============================

def pad_with_nan(
    array:np.ndarray,
    target_length:int,
    axis:int = 0,
    both_side:bool = False,
) -> np.ndarray:
    """
    Rellena un array con NaNs para alcanzar una longitud objetivo en un eje determinado.

    Args:
        array (np.ndarray): Array de entrada de tipo float16, float32 o float64.
        target_length (int): Longitud deseada a lo largo del eje especifciado.
        axis (int): Eje a lo largo del cual se realizará el padding.
        both_side (bool): True para repartir el padding en ambos lados y False para
            añadir los NaNs al final.
    
    Returns:
        np.ndarray: Array con padding de NaNs aplicado hasta alcanzar la longitud
            deseada.
    """
    # Comprueba que el tipo del array sea correcto.
    assert array.dtype in [np.float16, np.float32, np.float64]

    # Calcula cuanto padding se necesita.
    pad_size:int = target_length - array.shape[axis]

    # Retorna el array si la longitud es mayor que la deseada.
    if pad_size <= 0: return array

    # Inicializa la lista de tuplas de padding para cada eje.
    npad = [(0, 0)] * array.ndim

    # Configura el padding según si se reparte en ambos lados o solo al final.
    if both_side:
        # Divide el padding entre inicio y fin.
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        # Agrega todo el padding al final.
        npad[axis] = (0, pad_size)
    
    # Aplica el padding con NaNs y devuelve el array.
    return np.pad(
        array=array,
        pad_width=npad,
        mode="constant",
        constant_values=np.nan
    )


def split_with_nan(
    x:np.ndarray,
    sections:int,
    axis:int = 0  
) -> List[np.ndarray]:
    """
    Divide un array en secciones iguales a lo largo de un eje, rellenando con
    NaNs para que todas las secciones tengan la misma longitud.

    Args:
        x (np.ndarray): Array de entrada de tipo float16, float32 o float64.
        sections (int): Número de secciones en las que dividir el array.
        axis (int): Eje a lo largo del cual dividir el array.
    
    Returns:
        List[np.ndarray]: Lista de arrays, cada uno con u npadding de NaNs si es
            necesario para igualar la longitud de todas las secciones.
    """
    # Comprueba que el tipo del array sea correcto.
    assert x.dtype in [np.float16, np.float32, np.float64]

    # Separa en varios arrays.
    arrs:List[np.ndarray] = np.array_split(
        ary=x, 
        indices_or_sections=sections, 
        axis=axis
    )

    # Obtiene la longitud de la primera sección como referencia para le padding.
    target_length = arrs[0].shape[axis]

    # Recorre cada sección y aplica padidng con NaNs para igualar la longitud.
    for i in range(len(arrs)):
        # Aplica padding.
        arrs[i] = pad_with_nan(
            array=arrs[i],
            target_length=target_length,
            axis=axis
        )

    # Returns.
    return arrs

def take_per_row(
    A:np.ndarray|torch.Tensor,
    indx:np.ndarray|torch.Tensor,
    num_elements:int
) -> np.ndarray|torch.Tensor:
    """
    Extrae, para cada fila de una matriz A, un bloque consecutivo de elementos
    comenzando en la posición indicada por `indx`.

    Args:
        A (np.ndarray|torch.Tensor): Matriz 2D de la que se extraen valores.
        indx (np.ndarray|torch.Tensor): Índices iniciales (uno por fila) desde 
            donde comenzar a extraer `num_elemens` elementos.
        num_elements (int): Número de elementos consecutivos a extraer a partir 
            de cada índice.

    Returns:
        np.ndarray|torch.Tensor: Tensor/matriz con shape (n_filas, num_elements).
            Cada fila contiene los elementos extraídos de A según los índices indicados
            en `indx`.
    """
    # Construye una matriz de índices por fila.
    all_index = indx[:, None] + np.arange(num_elements)
    # Selecciona valores de A usando indexado avanzado.
    return A[torch.arange(all_index.shape[0])[:, None], all_index]

def centerize_vary_length_series(
        x:torch.Tensor
) -> torch.Tensor:
    """
    Centra series temporales con longitudes variables desplazando cada serie para
    que la parte válida (no NaN) quede centrada en el eje temporal.

    Args:
        x (np.ndarray): Tensor de forma (B, T, C), donde cada serie puede tener regiones
            iniciales o finales llenas de NaNs.
    
    Returns:
        np.ndarray: El mismo tensor, pero con cada serie centrada temporalmente.
    """
    # Busca para cada serie, el primer timestap que no es NaN.
    prefix_zeros:np.ndarray = np.argmax(
        a=~np.isnan(x).all(axis=-1),
        axis=1
    )

    # Hace lo mismo pero mirando la serie al revés.
    suffix_zeros:np.ndarray = np.argmax(
        a=~np.isnan(x[:, ::-1]).all(axis=-1),
        axis=1
    )

    # Calcula cuántas posiciones hay que mover la serie para que la parte con datos quede
    # en el centro.
    offset:np.ndarray = (prefix_zeros + suffix_zeros) // 2 - suffix_zeros

    # Crea índices de fila.
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]

    # Si el offset es negativo, se ajusta sumando T, de forma que el desplazamiento
    # funcione con indexado modular.
    # Esto evita índices negativos invalidando la rotación circular.
    offset[offset < 0] += x.shape[1]

    # Ajusta la matriz de índices temporales restando el offset de cada serie.
    # Genera un indezado circular que desplaza la parte válida al centro.
    column_indices = column_indices - offset[:, np.newaxis]

    # Devuelve por reindexado.
    return x[rows, column_indices]