# ==========================================================================================
# Author: Pablo González García.
# Created: 02/11/2025
# Last edited: 02/12/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List, Callable
# Interno:
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
# Internos:
from model import TS2VecModel
from loss import hierarchical_contrastive_loss


# ==============================
# FUNCIONES
# ==============================

def take_per_row(
    A:np.ndarray|torch.Tensor,
    indx:np.ndarray,
    num_elements:int
) -> np.ndarray|torch.Tensor:
    """
    Extrae, para cada fila de una matriz A, un bloque consecutivo de elementos
    comenzando en la posición indicada por `indx`.

    Args:
        A (np.ndarray|torch.Tensor): Matriz 2D de la que se extraen valores.
        indx (np.ndarray): Índices iniciales (uno por fila) desde 
            donde comenzar a extraer `num_elements` elementos.
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


# ==============================
# CLASES
# ==============================

class TS2VecTrainer:
    """
    Gestor del ciclo de entrenamiento para TS2Vec. Se encarga de ejecutar el
    bucle de entrenamiento, gestionar optimizadores, calcular pérdidas
    y ejecutar callbacks. Aísla la lógica de entrenamiento del modelo.
    """
    # ---- Default ---- #

    def __init__(
        self,
        model:TS2VecModel,
        optimizer:optim.Optimizer,
        device:str = 'cuda',
        temporal_unit:int = 0,
        max_train_length:int|None = None
    ) -> None:
        """
        Inicializa el entrenador.

        Args:
            model (nn.Module): Modelo a entrenar.
            optimizer (Optimizer): EL optimizador configurado.
            device (str): Dispositivo de ejecución ('cpu' o 'cuda').
            temporal_unit (int): La unidad mínima para realizar contraste temporal. Este parámetro ayuda a reducir el coste
                de tiempo y memoria para secuencias muy largas.
            max_train_length (int|None): La máxima longitud de la secuencia para entrenamiento. Secuencias mayores
                que este valor serán separadas en subsecuencias donde cada una tendrá una longitud menor que
                `max_train_length`.
        """
        # Inicializa las propiedades.
        self.ts2vec:TS2VecModel = model
        self.optimizer:optim.Optimizer = optimizer
        self.device:str = device
        self.temporal_unit:int = temporal_unit
        self.max_train_length:int|None = max_train_length

        self.n_epochs:int = 0
        self.n_iters:int = 0
    

    # ---- Métodos ---- #

    def fit(
        self,
        train_loader:DataLoader,
        n_epochs:int|None = None,
        n_iters:int|None = None,
        verbose:bool = False,
        after_iter_callback:Callable|None = None,
        after_epoch_callback:Callable|None = None
    ) -> List[float]:
        """
        Ejecuta el entrenamiento completo.

        Args:
            train_loader (DataLoader): Cargador de datos de entrenamiento.
            n_epochs (int): Número de épocas.
            n_iters (int): Númer de iteraciones.
            verbose (bool): Si e True, se muestra información al terminar cada época.
            after_iter_callback (Callable|None): Función a ejecutar tras cada iteración.
            after_epoch_callbakc (Callable|None): Función a ejecutar tras cada época.
        
        Returns:
            List[float]: Lista con la pérdida promedio de cada época.
        """
         # Listado para almacenar el loss promedio por época.
        losses:List[float] = []

        # Bucle de entrenamiento principal.
        while True:
            # Termina si alcanza el número máximo de épocas.
            if n_epochs is not None and self.n_epochs >= n_epochs: break

            # Variable para acumular el loss de una época.
            cum_loss:float = 0.0
            # Variable para contar el númeor de iteraciones por época.
            n_epoch_iters:int = 0
            # Flag para interrumpir el entrenamiento si se alcanza el máximo de iteraciones.
            interupted:bool = False

            # Bucle por batch.
            for batch in train_loader:
                # Termina si alcanza el número máximo de iteraciones.
                if n_iters is not None and self.n_iters >= n_iters:
                    interupted = True
                    break
                
                # Obtiene los datos del batch.
                x:torch.Tensor = batch[0]

                # Si la secuencia es demasiado larga, se toma una subventana aleatoria.
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    # Calcula el offset de la ventana aleatoriamente.
                    windows_offset:int = np.random.randint(low=x.size(1) - self.max_train_length + 1)
                    x = x[:, windows_offset : windows_offset + self.max_train_length]

                # Envía los datos al dispositivo.
                x = x.to(device=self.device)

                # Calcula la longitud de la serie temporal del batch.
                ts_l:int = x.size(1)

                # Define recortes aleatorios para la pérdida contrastiva jerárquica.
                crop_length:int = np.random.randint(
                    low=2 ** (self.temporal_unit + 1),
                    high=ts_l + 1
                )
                crop_left:int = np.random.randint(low=ts_l - crop_length + 1)
                crop_right:int = crop_left + crop_length
                crop_e_left:int = np.random.randint(low=crop_left + 1)
                crop_e_right:int = np.random.randint(
                    low=crop_right,
                    high=ts_l + 1
                )
                crop_offset = np.random.randint(
                    low=-crop_e_left,
                    high=ts_l - crop_e_right + 1,
                    size=x.size(0)
                )

                # Reinicia los gradientes del optimizador.
                self.optimizer.zero_grad()

                # Aplica la red a la primera vista recortada de la serie.
                out_1:torch.Tensor = self.ts2vec.encoder(take_per_row(
                    A=x,
                    indx=crop_offset * crop_e_left,
                    num_elements=crop_right - crop_e_left
                ))[:, -crop_length:]
                # Aplica la red a la segunda vista recortada de la serie.
                out_2:torch.Tensor = self.ts2vec.encoder(take_per_row(
                    A=x,
                    indx=crop_offset * crop_left,
                    num_elements=crop_e_right - crop_left
                ))[:, :crop_length]

                # Calcula la pérdida contrastiva jerárquica.
                loss:torch.Tensor = hierarchical_contrastive_loss(
                    z1=out_1,
                    z2=out_2,
                    temporal_unit=self.temporal_unit
                )

                # Backpropagatin.
                loss.backward()
                self.optimizer.step()
                # Sincroniza parámetros.
                self.ts2vec.model.update_parameters(self.ts2vec.encoder)

                # Acumula loss e iteraciones.
                cum_loss += loss.item()
                n_epoch_iters += 1
                # Incrementa el contador de iteraciones.
                self.n_iters += 1

                # Callback opcional después de cada iteracción.
                if after_iter_callback is not None: after_iter_callback(self.n_iters, loss.item())
            
            # Termina completamente si ha sido interrumpido por alcanzar el máximo de iteracciones.
            if interupted: break

            # Calcula el loss promedio de la época.
            cum_loss /= n_epoch_iters
            losses.append(cum_loss)

            # Incrementa el contador de épocas.
            self.n_epochs += 1

            # Callback opcional después de cada época.
            if after_epoch_callback is not None: after_epoch_callback(self.n_epochs, cum_loss)

            # Imprime si verbose es True.
            if verbose: print(f"Epoch ({self.n_epochs}):\tloss={cum_loss}")

        return losses