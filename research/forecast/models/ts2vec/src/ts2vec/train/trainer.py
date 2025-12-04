# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 03/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List, Callable
# Externos:
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
# Internos:
from .utils import take_per_row


# ==============================
# CLASES
# ==============================

class TS2VecTrainer:
    """
    Clase encargada de ejecutar el bucle de entrenamiento, gestionar optimizadores,
    calcular pérdidas y ejecutar callbacks del entrenamiento.
    """
    # ---- Default ---- #

    def __init__(
        self,
        optimizer:optim.Optimizer,
        device:str = 'cuda'
    ) -> None:
        """
        Inicializa el entrenador.

        Args:
            optimizer (optim.Optimizer): Optimizador del entrenamiento.
            device (str): Dispositivo de ejecución. Puede ser `cpu` o `cuda`.
        """
        # Inicializa las propiedades.
        self.optimizer:optim.Optimizer = optimizer
        self.device:str = device
        self.n_iters:int = 0
        self.n_epochs:int = 0
    

    # ---- Métodos ---- #

    def train(
        self,
        data:DataLoader,
        n_iters:int|None = None,
        n_epochs:int|None = None,
        temporal_unit:int = 1,
        after_epoch_callback:Callable|None = None,
        after_iter_callback:Callable|None = None,
        verbose:bool = False
    ):
        """
        Entrena el modelo.

        Args:
            data (DataLoader): Cargador de datos de entrenamiento.
            n_iters (int): Número de iteraciones del entrenamiento.
            n_epochs (int): Número de épocas del entrenamiento.
            temporal_unit (int): 
            after_epoch_callback (Callable|None): Función a ejecutar tras completar
                una época.
            after_iter_callback (Callable|None): Función a ejecutar tras completar
                una iteración.
            verbose (bool): Si es True, se muestra información al terminar cada época.
        """
        # Listado para almacenar las losses del entrenamiento.
        losses:List[float] = []

        # Bucle del entrenamiento.
        while True:
            # Finaliza el búcle si se ejecutáron todas las épocas.
            if n_epochs is not None and self.n_epochs >= n_epochs: break

            # Almacena la loss acumulada.
            cum_loss:float = 0.0
            # Indica si se ha interrumpido por llegar al final.
            interrupted:bool = False
            # Indica el número de iteraciones realizadas en cada época.
            n_epoch_iters:int = 0

            # Itera sobre los batches.
            for batch in data:
                # Finaliza el búcle si se alcanza el número máximo de iteraciones.
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                # Obtiene los datos del batch.
                x:torch.Tensor = batch[0]

                # Envía los datos al dispositivo.
                x.to(device=self.device)

                # Calcula la longitud de la serie temporal.
                ts_l:int = x.shape[1]

                # Define recortes aleatorios para la pérdida contrastiva jerárquica.
                crop_length:int = np.random.randint(
                    low=2 ** (temporal_unit + 1),
                    high=ts_l + 1
                )
                crop_left:int = np.random.randint(low=ts_l - crop_length + 1)
                crop_right:int = crop_left + crop_length
                crop_e_left:int = np.random.randint(low=crop_left + 1)
                crop_e_right:int = np.random.randint(
                    low=crop_right,
                    high=ts_l + 1
                )
                crop_offset:np.ndarray = np.random.randint(
                    low=crop_e_left,
                    high=ts_l - crop_e_right + 1,
                    size=x.size(dim=0)
                )

                # Reinicia los gradientes del optimizador.
                self.optimizer.zero_grad()

                # TODO: Aplica la red a la primera y segunda vista.

                # TODO: Calcúla la pérdida contrastiva.

                # TODO: Backpropagation para el gradiente.


                # Incrementa el número de iteraciones.
                self.n_iters += 1
                n_epoch_iters += 1

                # Callback opcional despúes de cada iteración.
                if after_iter_callback is not None: after_iter_callback(self.n_iters)

            # Calcula el loss promedio de la época.
            cum_loss /= n_epoch_iters
            losses.append(cum_loss)

            # Incrementa el número de épocas.
            self.n_epochs += 1

            # Callback opcional después de cada época.
            if after_epoch_callback is not None: after_epoch_callback(self.n_epochs)

            # Imprime si verbose es True.
            if verbose: print(f"Epoch {self.n_epochs}")

            # Termina completamente si se han alcanzado el número de iteraciones.
            if interrupted: break