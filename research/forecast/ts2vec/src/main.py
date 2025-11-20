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
from typing import Callable
# Externos:
import torch
from torch.optim import swa_utils
# Internos:
from .encoder import TSEncoder


# ==============================
# CLASES
# ==============================

class TS2Vec:
    """
    
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dims:int,
        output_dims:int,
        hidden_dims:int = 64,
        depth:int = 10,
        device:str = 'cuda',
        learning_rate:float = 0.001,
        batch_size:int = 16,
        max_train_length:int|None = None,
        temporal_unit:int = 0,
        after_iter_callback:Callable|None = None,
        after_epoch_callback:Callable|None = None
    ) -> None:
        """
        Inicializa el TS2Vec.

        Args:
            input_dims (int): La dimensión del input. Para series temporales univariantes, este
                valor debe ser 1.
            output_dims (int): La dimensión de la representación.
            hidden_dims (int): La dimesión oculta del encoder.
            depth (int): El número de bloques residuales del encoder.
            device (int): Dispositivo de ejecución.
            learning_rate(float): El ratio de aprendizaje.
            batch_size (int): El tamaño del batch.
            max_train_length (int): La máxima longitud de la secuencia para entrenamiento. Secuencias mayores
                que este valor seran separadas en subsecuencias donde cada una tendrá una longitud menor que
                `max_train_length`.
            temporal_unit (int): La unidad mínima para realizar contraste temporal. Este parámetro ayuda a reducir el coste
                de tiempo y memoria para secuencias muy largas.
            after_iter_callback (Callable|None): Función llamada después de cada iteración.
            after_epoch_callback (Callable|None): Función llamada después de cada época.
        """
        # Inicializa las propiedades.
        self.device:str = device
        self.learning_rage:float = learning_rate
        self.batch_size:int = batch_size
        self.max_train_length:int|None = max_train_length
        self.temporal_unit:int = temporal_unit
        self.after_iter_callback:Callable|None = after_iter_callback
        self.after_epoch_callback:Callable|None = after_epoch_callback
        self.n_epochs:int = 0
        self.n_iters:int = 0

        # Inicializa el encoder de la serie temporal.
        self._net:TSEncoder = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth
        ).to(device=self.device)

        self.net:swa_utils.AveragedModel = swa_utils.AveragedModel(model=self._net)
        self.net.update_parameters(self._net)
    

    # ---- Métodos ---- #

    def save(
        self,
        file_name:str
    ) -> None:
        """
        Guarda el modelo en un fichero.

        Args:
            file_name (str): Nombre del fichero.
        """
        # Almacena el fichero.
        torch.save(
            obj=self.net.state_dict(),
            f=file_name
        )
    
    def load(
        self,
        file_name:str
    ) -> None:
        """
        Carga el modelo de un fichero.

        Args:
            file_name (str): Nombre del fichero.
        """
        # Carga la información del modelo.
        state_dict = torch.load(
            f=file_name,
            map_location=self.device
        )
        # Carga los parámetros.
        self.net.load_state_dict(state_dict=state_dict)