# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 04/12/2025
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
from train.utils import take_per_row
from train.loss import hierarchical_contrastive_loss
from core.encoder import SWAEncoder


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
        model:SWAEncoder,
        optimizer:optim.Optimizer,
        device:str = 'cuda'
    ) -> None:
        """
        Inicializa el entrenador.

        Args:
            model (SWAEncoder): Modelo a entrenar.
            optimizer (optim.Optimizer): Optimizador del entrenamiento.
            device (str): Dispositivo de ejecución. Puede ser `cpu` o `cuda`.
        """
        # Inicializa las propiedades.
        self.model:SWAEncoder = model
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
    ) -> List[float]:
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
        
        Returns:
            List[float]: Lista con las pérdidas obtenidas para cada época.
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
                x = x.to(device=self.device)

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
                    low=-crop_e_left,
                    high=ts_l - crop_e_right + 1,
                    size=x.size(dim=0)
                )

                # Reinicia los gradientes del optimizador.
                self.optimizer.zero_grad()

                # Aplica la red a la primera y segunda vista.
                out1:torch.Tensor = self.model(
                    take_per_row(
                        A=x,
                        indx=crop_offset + crop_e_left,
                        num_elements=crop_right - crop_e_left
                    )
                )[:, -crop_length:]
                out2:torch.Tensor = self.model(
                    take_per_row(
                        A=x,
                        indx=crop_offset + crop_left,
                        num_elements=crop_e_right - crop_left
                    )
                )[:, :crop_length]

                # Calcúla la pérdida contrastiva.
                loss:torch.Tensor = hierarchical_contrastive_loss(
                    z1=out1,
                    z2=out2,
                    temporal_unit=temporal_unit
                )

                # Backpropagation para el gradiente.
                loss.backward()
                self.optimizer.step()
                # Actualiza los parámetros del modelo.
                self.model.update()

                # Acumula la loss.
                cum_loss += loss.item()

                # Incrementa el número de iteraciones.
                self.n_iters += 1
                n_epoch_iters += 1

                # Callback opcional despúes de cada iteración.
                if after_iter_callback is not None: after_iter_callback(self.n_iters, cum_loss)

            # Termina completamente si se han alcanzado el número de iteraciones.
            if interrupted: break
            
            # Calcula el loss promedio de la época.
            cum_loss /= n_epoch_iters
            losses.append(cum_loss)

            # Incrementa el número de épocas.
            self.n_epochs += 1

            # Callback opcional después de cada época.
            if after_epoch_callback is not None: after_epoch_callback(self.n_epochs, cum_loss)

            # Imprime si verbose es True.
            if verbose: print(f"Epoch {self.n_epochs}:\tLoss: {cum_loss}")
        
        return losses