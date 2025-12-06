# ==========================================================================================
# Author: Pablo González García.
# Created: 04/12/2025
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
from torch.utils.data import TensorDataset, DataLoader
# Internos:
from core.encoder import SWAEncoder, TSEncoder
from train.trainer import TS2VecTrainer
from train.utils import centerize_vary_length_series, split_with_nan


# ==============================
# CLASES
# ==============================

class TS2Vec:
    """
    Modelo de TS2Vec.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        hidden_dim:int = 64,
        depth:int = 10,
        device:str = 'cuda:0',
        batch_size:int = 16
    ) -> None:
        """
        Inicializa el TS2Vec.

        Args:
            input_dim (int): La dimensión del input. Para series temporales univariantes, este
                valor debe ser 1.
            output_dim (int): La dimensión de la representación.
            hidden_dim (int): La dimensión oculta del encoder.
            depth (int): El número de bloques residuales del encoder.
            device (int): Dispositivo de ejecución.
            batch_size (int): El tamaño del batch.
        """
        # Inicializa las propiedades.
        self.device:str = device
        self.batch_size:int = batch_size
        self.n_epochs:int = 0
        self.n_iters:int = 0

        # Inicializa el encoder de la serie temporal.
        self.model:SWAEncoder = SWAEncoder(
            encoder=TSEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                depth=depth
            ),
            device=self.device
        )
    

    # ---- Métodos ---- #

    def fit(
        self,
        train_data:np.ndarray,
        n_epochs:int|None = None,
        n_iters:int|None = None,
        max_train_length:int|None = None,
        learning_rate:float = 0.001,
        temporal_unit:int = 0,
        after_iter_callback:Callable|None = None,
        after_epoch_callback:Callable|None = None,
        verbose:bool = False
    ) -> List[float]:
        """
        Entrena el modelo usando los datos proporcionados.

        Args:
            train_data (np.ndarray): Los datos de entrenamiento, con la forma (n_instance, n_timestamps, n_features).
                Los valores faltantes deben ser NaN.
            n_epochs (int): El número máximo de épocas. Se detiene el entrenamiento al alcanzarlo.
            n_iters (int): El número máximo de iteraciones. Se detiene el entrenamiento al alcanzarlo.
            max_train_length (int): La máxima longitud de la secuencia para entrenamiento. Secuencias mayores
                que este valor serán separadas en subsecuencias donde cada una tendrá una longitud menor que
                `max_train_length`
            learning_rate (float): El ratio de aprendizaje.
            temporal_unit (int): La unidad mínima para realizar contraste temporal. Este parámetro ayuda a reducir el coste
                de tiempo y memoria para secuencias muy largas.
            after_iter_callback (Callable|None): Función llamada después de cada iteración.
            after_epoch_callback (Callable|None): Función llamada después de cada época.
            verbose (bool): Si es True, imprime el loss promedio por época.

        Returns:
            List[float]: Lista de losses promedio por época.
        """
         # Verifica que los datos de entrenamiento sean un array 3D.
        assert train_data.ndim == 3

        # Establece el número de iteraciones por defecto.
        if n_iters is None and n_epochs is None: n_iters = 200 if train_data.size <= 100000 else 600

        # Si existe un límite de longitud de entrenamiento, se divide la serie.
        if max_train_length is not None:
            # Separa el conjunto de datos.
            sections:int = train_data.shape[1] // max_train_length
            # Si hay más de una sección, se concatenan.
            if sections >= 2: train_data = np.concatenate(
                split_with_nan(
                    x=train_data,
                    sections=sections,
                    axis=1
                ),
                axis=0
            )
        
        # Detecta si hay timestamps completamente faltantes al inicio o al final.
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:                                 # type: ignore
            train_data = centerize_vary_length_series(x=train_data)
        # Elimina series o instancias completamente NaN.
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]          # type: ignore

        # Convierte los datos en un TensorDataset de PyTorch.
        train_dataset:TensorDataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader:DataLoader = DataLoader(
            dataset=train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True      # Descarta batch incompleto.
        )

        # Inicializa el optimizador AdamW para entrenar la red.
        optimizer:torch.optim.AdamW = torch.optim.AdamW(
            self.model.core.parameters(),
            lr=learning_rate
        )

        # Genera el entrenador.
        trainer:TS2VecTrainer = TS2VecTrainer(
            model=self.model,
            optimizer=optimizer,
            device=self.device
        )
        # Ejecuta el entrenamiento.
        return trainer.train(
            data=train_loader,
            n_iters=n_iters,
            n_epochs=n_epochs,
            temporal_unit=temporal_unit,
            after_iter_callback=after_iter_callback,
            after_epoch_callback=after_epoch_callback,
            verbose=verbose
        )