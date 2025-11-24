# ==========================================================================================
# Author: Pablo González García.
# Created: 24/11/2025
# Last edited: 24/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import time
import datetime
from argparse import ArgumentParser, Namespace
# Internos:
import console
from main import TS2Vec


# ==============================
# CONSTANCES
# ==============================

DEFAULT_BATCH_SIZE:int = 8
DEFAULT_LEARNING_RATE:float = 0.001
DEFAULT_REPR_DIMS:int = 320
DEFAULT_ITERS:int = 5000
DEFAULT_EPOCHS:int = 2


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

    # Retorna los argumentos.
    return parser.parse_args()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except para manejo de errores.
    try:
        # Imprime el título.
        console.print_h1(
            text="ENTRENAMIENTO",
            spacing_up=1,
            spacing_down=1
        )

        # Comprueba y convierte los argumentos.
        args:Namespace = manage_arguments()

        # Inicializa el modelo.
        model:TS2Vec = TS2Vec(
            input_dims=1,
            output_dims=args.repr_dims
        )

        # Imprime información de ejecución.
        console.info(text="Entrenamiento iniciado.")

        # Obtiene el tiempo inicial.
        start_time:float = time.time()

        # Obtiene el tiempo final.
        end_time:float = time.time() - start_time

        # Imprime información de ejecución.
        console.info(text="Entrenamiento finalizado.")
        
        # Imprime información del entrenamiento.
        console.print_attribute(
            key="Duración del entrenamiento",
            value=datetime.timedelta(seconds=end_time)
        )
        console.print_attribute(
            key="Número de épocas ejecutadas",
            value=0
        )
        console.print_attribute(
            key="Número de iteraciones ejecutadas",
            value=0
        )
        console.print_attribute(
            key="Loss final",
            value=0.00
        )

    # Si ocurre algún error.
    except Exception as ex:
        # Imrpime el error.
        console.error(f"Error durante el entrenamiento: {ex}")