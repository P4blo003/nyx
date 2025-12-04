# ==========================================================================================
# Author: Pablo González García.
# Created: 24/11/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import os
import time
from typing import List
from datetime import timedelta, datetime
from argparse import ArgumentParser, Namespace
# Externos:
import torch
import numpy as np
# Internos:
import core.dl as dl
import utils.dataset as dataset
from api.model import TS2Vec


# ==============================
# CONSTANTES
# ==============================

DEFAULT_GPU:int = 0
DEFAULT_SEED:int|None = None
DEFAULT_MAX_NUM_THREADS:int = 8
DEFAULT_BATCH_SIZE:int = 8
DEFAULT_LR:float = 0.001
DEFAULT_REPR_DIM:int = 320
DEFAULT_ITERS:int|None = None
DEFAULT_EPOCHS:int|None = None
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
    parser.add_argument(
        'dataset',
        type=str,
        help="Ruta al fichero del dataset."
    )

    # Argumentos opcionales.
    parser.add_argument(
        '--gpu',
        type=int,
        default=DEFAULT_GPU,
        help=f"El número de la GPU usada para entrenamiento e inferencia (Por defecto es {DEFAULT_GPU})."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Semilla aleatoria."
    )
    parser.add_argument(
        '--max-threads',
        type=int,
        default=DEFAULT_MAX_NUM_THREADS,
        help=f"Máximo número de hilos permitidos usados por este proceso (Por defecto es {DEFAULT_MAX_NUM_THREADS})."
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Tamaño del batch (Por defecto es {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_LR,
        help=f"Ratio de aprendizaje (Por defecto es {DEFAULT_LR})"
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=DEFAULT_ITERS,
        help=f"Número de iteraciones. (Por defecto es {DEFAULT_ITERS})"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Número de épocas (Por defecto es {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        '--reprs-dim',
        type=int,
        default=DEFAULT_REPR_DIM,
        help=f"Dimensión de los embeddings generados (Por defecto es {DEFAULT_REPR_DIM})."
    )
    parser.add_argument(
        '--max-train-length',
        type=int,
        default=DEFAULT_MAX_TRAIN_LENGTH,
        help=f"Número de épocas (Por defecto es {DEFAULT_MAX_TRAIN_LENGTH})"
    )

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


        # ---- Inicialización ---- #

        # Imprime información.
        print("Inicializando recursos ...")

        # Inicializa el dispositivo de ejecución.
        device:List[torch.device]|torch.device = dl.init_dl_program(
            device_name=args.gpu,
            seed=args.seed,
            max_threads=args.max_threads
        )

        # Almacena el momento de la ejecución.
        run_date:str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        # Inicializa el nombre donde se almacenará toda la información de entrenamiento.
        training_dir:str = f"training/{run_date}"
        # Crea el directorio.
        os.makedirs(name=training_dir, exist_ok=True)

        # Imprime información.
        print("Recursos inicializados.")
        

        # ---- Preprocesamiento ---- #

        # Imprime información.
        print("Procesando los datos ...")

        # Carga el conjunto de datos.
        data, train_slice, valid_slice, test_slice, scaler, pred_lens = dataset.load_csv(
            dataset=args.dataset,
            separator=',',
            has_header=True,
            timestamp_column='Timestamp',
            timestamp_format="%d/%m/%Y %H:%M",
            categorical_columns=['WS', 'DW', 'LT']
        )
        # Crea los conjuntos de entrenamiento, validación y test.
        train:np.ndarray = data[:, train_slice, :]
        valid:np.ndarray = data[:, valid_slice, :]
        test:np.ndarray = data[:, test_slice, :]

        # Imprime la información.
        print("Datos procesados.")
        print(f"Dataset entrenamiento: {train.shape}")
        print(f"Dataset validación: {valid.shape}")
        print(f"Dataset prueba: {test.shape}")


        # ---- Entrenamiento ---- #

        # Inicializa el modelo.
        model:TS2Vec = TS2Vec(
            input_dim=train.shape[-1],
            output_dim=args.reprs_dim,
            device=str(device),
            batch_size=args.batch_size
        )

        # Obtiene el tiempo inicial.
        start:float = time.time()

        # Entrena el modelo.
        losses:List[float] = model.fit(
            train_data=train,
            n_iters=args.iters,
            n_epochs=args.epochs,
            verbose=True
        )

        # Obtiene la duración del entrenamiento.
        duration:float = time.time() - start

        # Imprime la información.
        print(f"Entrenamiento lanzado ({run_date}) finalizado. Duración: {timedelta(seconds=duration)}")

    # Si ocurre algún error.
    except Exception as ex:
        # Imprime el error.
        print(f"{type(ex)} => {ex}")
        # Relanza el error.
        raise ex