# ==========================================================================================
# Author: Pablo González García.
# Created: 24/11/2025
# Last edited: 25/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import time
import datetime
from argparse import ArgumentParser, Namespace
# Externos:
import polars as pl
import numpy as np
# Internos:
import console
from preprocessing import Preprocessor
from main import TS2Vec


# ==============================
# CONSTANCES
# ==============================

DEFAULT_BATCH_SIZE:int = 64
DEFAULT_LEARNING_RATE:float = 0.001
DEFAULT_REPR_DIMS:int = 320
DEFAULT_ITERS:int = 2500
DEFAULT_EPOCHS:int = 75
DEFAULT_NAME:str = 'embedder'


# ==============================
# FUNCIONES
# ==============================

def manage_arguments() -> Namespace:
    """
    Comprueba, convierte y devuelve los argumentos del programa.

    Returns:
        Namespace: Argumentos convertidos.
    """
    # Inicializa el parseador de argumentos.
    parser:ArgumentParser = ArgumentParser()

    # Establece los argumentos del programa.
    parser.add_argument(
        'dataset',
        help="Ruta del fichero con los datos."
    )
    parser.add_argument(
        '--name',
        type=str,
        default=DEFAULT_NAME,
        help=f"Nombre del modelo. Se emplea para el nombre del fichero. Por defecto es {DEFAULT_NAME}."
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"El tamaño de cada batch. Por defecto es {DEFAULT_BATCH_SIZE}."
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"El ratio de aprendizaje. Por defecto es {DEFAULT_LEARNING_RATE}."
    )
    parser.add_argument(
        '--repr-dims',
        type=int,
        default=DEFAULT_REPR_DIMS,
        help=f"La dimensión de la representación. Por defecto es {DEFAULT_REPR_DIMS}."
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=DEFAULT_ITERS,
        help=f"Número de iteraciones. Por defecto es {DEFAULT_ITERS}."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Número de épocas. Por defecto es {DEFAULT_EPOCHS}."
    )
    parser.add_argument(
        '--save',
        type=bool,
        default=True,
        help="Si el modelo será guardado tras finalizar el entrenamiento."
    )
    parser.add_argument(
        '--plot',
        type=bool,
        default=True,
        help="Si se generará información adicional con ficheros y gráficas."
    )

    # Retorna los argumentos.
    return parser.parse_args()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except para manejo de errores.
    try:
        # Comprueba y convierte los argumentos.
        args:Namespace = manage_arguments()

        # Imprime cabecera de la sección.
        console.print_h1(
            text="Preprocesamiento",
            spacing_up=1,
            spacing_down=1
        )

        # TODO: Carga los datos del fichero.
        df:pl.DataFrame = pl.read_csv(
            source=args.dataset,                # Nombre del dataset.,
            separator=',',                      # Separador de los datos.
            has_header=True,                    # Indica que tiene cabeceras.
            infer_schema_length=1000            # Usa 1000 entradas para diferir los datos del csv.
        )

        # Genera el preprocesador para el dataset.
        preprocessor:Preprocessor = Preprocessor()
        
        # Preprocesar los datos.
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, num_ts_features  = preprocessor(
            df=df,
            timestamp_column="Timestamp",
            timestamp_format="%d/%m/%Y %H:%M",
            categorical_columns=['WS', 'DW', 'LT']
        )
        # Genera los conujuntos para el modelo.
        x_train:np.ndarray = data[:, train_slice, :]
        x_valid:np.ndarray = data[:, valid_slice, :]
        x_test:np.ndarray = data[:, test_slice, :]

        # Imprime información de ejecución.
        console.info(text="Preprocesamiento finalizado.")

        # Imprime información del preprocesamiento.
        console.print_attribute(
            key="Dataset",
            value=data.shape
        )
        console.print_attribute(
            key="Dataset de entrenamiento",
            value=x_train.shape
        )
        console.print_attribute(
            key="Dataset de validación",
            value=x_valid.shape
        )
        console.print_attribute(
            key="Dataset de test",
            value=x_test.shape
        )

        # Imprime cabecera de la sección.
        console.print_h1(
            text="Entrenamiento",
            spacing_up=1,
            spacing_down=1
        )

        # Inicializa el modelo.
        model:TS2Vec = TS2Vec(
            input_dims=x_train.shape[-1],
            output_dims=args.repr_dims,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        # Calcula el tiempo inicial.
        start:float = time.time()

        # Ejecuta el entrenamiento.
        model.fit(
            train_data=x_train,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True
        )

        # Obtiene el tiempo de entrenamiento.
        end:float = time.time() - start

        # Imprime información de ejecución.
        console.info(text="Entrenamiento finalizado.")

        # Imprime información del entrenamiento.
        console.print_attribute(
            key="Druación del entrenamiento",
            value=datetime.timedelta(seconds=end)
        )

    # Si ocurre algún error.
    except Exception as ex:
        # Imrpime el error.
        console.error(f"Error durante el entrenamiento: {ex}")
    
    # Se ejecuta al final.
    finally:
        # Imprime un salto de línea.
        print()