# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 03/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Externos:
import numpy as np
import torch


# ==============================
# FUNCIONES
# ==============================

def take_per_row(
    A:np.ndarray|torch.Tensor,
    indx:np.ndarray,
    num_elements:int
) -> np.ndarray|torch.Tensor:
    """
    Extrae para cada fila de una matriz A, un bloque consecutivo de elementos
    comenzando en la posición indicada por `indx`.

    Args:
        A (numpy.ndarray|torch.Tensor): Matriz 2D de la que se extrane valores.
        indx (numpy.ndarray): Índices iniciales (uno por fila) desde donde
            comenzar a extraer `num_elements` elementos.
        num_elements (int): Número de elementos consecutivos a extraer a partir
            de cada índice.
    
    Returns:
        np.ndarray|torch.Tensor: Tensor/matriz con forma (n_filas, num_elements).
            Cada fila contiene los elementos extraídos de A según los índices
            indicados en `indx`.
    """
    # Construye una matriz de índices por fila.
    all_indx:np.ndarray = indx[:, None] + np.arange(num_elements)
    # Selecciona valores de A usando indexado avanzado.
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]