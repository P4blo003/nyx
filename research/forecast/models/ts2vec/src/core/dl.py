# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import random
from typing import List
# Externos:
import torch
import numpy as np


# ==============================
# FUNCIONES
# ==============================

def init_dl_program(
    device_name,
    seed,
    use_cudnn:bool = True,
    deterministic:bool = False,
    benchmark:bool = False,
    use_tf32:bool = False,
    max_threads:int|None = None
) -> List[torch.device]|torch.device:
    """
    """
    # Configura el número máximo de hilos para librerías subyacentes (OpenMP, MKL).
    if max_threads is not None:
        # Hilos para operaciones internas.
        torch.set_num_threads(max_threads)
        # Comprueba si interop no coincide y lo ajusta explícitamente.
        if torch.get_num_interop_threads() != max_threads: torch.set_num_interop_threads(max_threads)

        # Try-Except para manejo de errores.
        try:
            # Importa el módulo.
            import mkl                      # type: ignore

        # Si MKL no está presente, no se realiza ninguna acción.
        except: pass
        # Si está presente, establece los hilos usados por MKL.
        else: mkl.set_num_threads(max_threads)

    # Inicializa todas las semillas para reproductibilidad (Python, numpy y torch).
    if seed is not None:
        # Establece la semilla para python.
        random.seed(a=seed)
        # Incrementa la semilla.
        seed += 1
        # Establece la semilla para numpy.
        np.random.seed(seed=seed)
        # Incrementa la semilla.
        seed += 1
        # Establece la semilla para torch.
        torch.manual_seed(seed=seed)

    # Permite pasar un único device o una lista de devices, normalizando la lista.
    if isinstance(device_name, (str, int)): device_name = [device_name]

    # Variable para almacenar dispositivos.
    devices:List[torch.device] = []

    # Recorre los dispositivos en orden inverso para asegurar set_device correcto.
    for t in reversed(device_name):
        # Obtiene el dispositivo.
        t_device:torch.device = torch.device(t)
        # Añade el dispositivo.
        devices.append(t_device)

        # Si el dispositivo es cuda, verifica disponibilidad y configura el contenido.
        if t_device.type == "cuda":
            # Verifica que esté disponible.
            assert torch.cuda.is_available()
            # Establece el dispositivo.
            torch.cuda.set_device(device=t_device)

            # Si se usa seed, inicializa la semilla del generador cuda.
            if seed is not None:
                # Incrementa la semilla.
                seed += 1
                # Establece la semilla.
                torch.cuda.manual_seed(seed=seed)
        
    # Restaura el orden original de los dispositivos.
    devices.reverse()

    # Configura parámetros globales de cudnn.
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    # Habilita o deshabilita precisión TF32 si torch lo soporta.
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    # Devuelve el dispositivo o lista de dispositivos en función del número recibido.
    return devices if len(devices) > 1 else devices[0]