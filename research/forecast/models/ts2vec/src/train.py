# ==========================================================================================
# Author: Pablo González García.
# Created: 24/11/2025
# Last edited: 27/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import os
import time
import datetime
from argparse import ArgumentParser, Namespace
# Externos:
import polars as pl
from preprocessing import Preprocessor
# Internos:
from ts2vec import TS2Vec


# ==============================
# CONSTANTES
# ==============================

DEFAULT_BATCH_SIZE:int = 8
DEFAULT_LEARNING_RATE:float = 0.001
DEFAULT_REPRESENTATION_DIM:int = 320
DEFAULT_MAX_TRAIN_LENGTH:int = 3000


# ==============================
# FUNCIONES
# ==============================

def manage_arguments() -> Namespace:
    """
    Gestiona los argumentos del programa.

    Returns:
        Namespace: Namespace con los argumentos dados.
    """
    # Inicializa el parser.
    parser:ArgumentParser = ArgumentParser()

    # Argumentos obligatorios.
    parser.add_argument('dataset', type=str, help="Ruta al fichero del dataset.")
    parser.add_argument('name', type=str, help="Nombre del entrenamiento, se emplea tanto para darle nombre al dataset como para el directorio donde se almacenan los resultados.")

    # Retorna el Namespace.
    return parser.parse_args()



# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except para manejo de errores.
    try:

        # ---- Argumentos ---- #

        # Carga los argumentos del programa.
        args:Namespace = manage_arguments()
    
    # Si ocurre algún error.
    except Exception as ex:
        # Imprime el error.
        print(f"{type(ex)} => {ex}")