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
from enum import Enum
# Externos:
import torch
import numpy as np


# ==============================
# ENUMS
# ==============================

class MaskMode(Enum):
    """
    
    """
    # ---- Atributos ---- #
    BINOMIAL    = 0
    CONTINUOUS  = 1
    ALL_TRUE    = 2
    ALL_FALSE   = 3
    MASK_LAST   = 4


# ==============================
# FUNCIONES
# ==============================

def generate_binomial_mask(
    B:int,
    T:int,
    p:float = 0.5
) -> torch.Tensor:
    """
    Genéra una mácara booleana aleatoria según una distribución binomial (Bernoulli)
    con probabilidad `p`.
    
    Args:
        B (int): Número de secuencias / ejemplos en el bacth (batch size).
        T (int): Longitud temporal de cada secuencia (número de timesteps).
        p (float): Probabilidad de que la máscara sea True en cada posición.
    
    Returns:
        torch.Tensor: Tensor booleano de forma `(B, T)` con valores True o False.
            Se emplea para enmascarar pasos temporales en modelos de series temporales.
    """
    # Genera una matriz numpy (B, T) con 0/1 siendo una distribución binomial con n=1.
    # Convierte el aray numpy a un tensor PyTorh y lo transofrma a booleano.
    return torch.from_numpy(np.random.binomial(
            n=1,
            p=p,
            size=(B, T)
        )
    ).to(dtype=torch.bool)

def generate_continuous_mask(
    B:int,
    T:int,
    n:int|float = 5,
    l:int|float = 0.1    
) -> torch.Tensor:
    """
    Genera una máscara booleana que enmascara segmetnos continuos. Cada secuencia del
    bacth tendrá `n` segmentos enmanscarados de longitud `l` cada uno.

    Args:
        B (int): Número de secuencias / ejemplos en el bacth (batch size).
        T (int): Longitud temporal de cada secuencia (número de timesteps).
        n (int): Número de segmentos a enmascarar por secuencia.
            - Si es un `int` se interpreta como número de bloques.
            - Si es un `float` se interpreta como una fracción de `T`.
        l (int|float): Longitud de cada segmento enmascarado.
            - Si es un `int` se interpreta como número de timestamps a enmascarar por bloque.
            - Si es un `float` se interpreta como una fracción de `T`.
    
    Returns:
        torch.Tensor: Mácara boolenaa de forma `(B, T)` con `False` en las posiciones enmascaradas
            (segmentos continuos generados aleatoriamente) y `True` en las restantes.
    """
    # Inicializa la máscara completamente verdadera (sin enmascaramiento).
    res = torch.full(
        size=(B, T),
        fill_value=True,
        dtype=torch.bool
    )

    # Si n es float, calcula el número de segmentos.
    if isinstance(n, float): n = int(n * T)
    # Se asegura de que almenos haya un segmento y como mucho T//2.
    n = max(min(n, T // 2), 1)

    # Si l es float, calcula el número de timestamps por segmento.
    if isinstance(l, float): l = int(l * T)
    # Al menos 1 timestep debe tener falso para que haya efecto de enmascaramiento.
    l = max(l, 1)

    # Para cada secuencia del bacth, dibuja n segmentos enmascarados.
    for i in range(B):
        for _ in range(n):
            # Elige aleatoriamente un punto del inicio entre [0, T-L]
            t = np.random.randint(T - l + 1)
            # Marca un bloque de longitud l como enmascarado.
            res[i, t:t + l] = False

    # Returns.
    return res