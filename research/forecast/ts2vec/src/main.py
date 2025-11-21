# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
# Last edited: 21/11/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import Callable, List
# Externos:
import numpy as np
import torch
from torch.optim import swa_utils
from torch.utils.data import TensorDataset, DataLoader
# Internos:
from .encoder import TSEncoder
from .math.loss import hierarchical_contrastive_loss
from .math.utils import split_with_nan, take_per_row


# ==============================
# CLASES
# ==============================

class TS2Vec:
    """
    Modelo de aprendizaje para la representación de series temporales basado en TS2Vec.
    Genera embeddings significativos de secuencias temporales, capturando patrones locales y globales,
    pudiendo usarse para tareas de clasificación, predicción, clustering o detección de
    anomalías.
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
    
    def fit(
        self,
        train_data:np.ndarray,
        n_epochs:int|None = None,
        n_iters:int|None = None,
        verbose:bool = False
    ) -> List[float]:
        """
        Entrena el modelo usando los datos proporcionados.

        Args:
            train_data (np.ndarray): Los datos de entrenamiento, con la forma (n_instance, n_timestamps, n_features).
                Los valores faltantes deben ser NaN.
            n_epochs (int): El número máximo de épocas. Se detiene el entrenamiento al alcanzarlo.
            n_iters (int): El número máximo de iteraciones. Se detiene el entrenamiento al alcanzarlo.
            verbose (bool): Si es True, imprime el loss promedio por época.
        
        Returns:
            List[float]: Lista de losses promedio por época.
        """
        # Verifica que los datos de enetrenamiento sean un array 3D.
        assert train_data.ndim == 3

        # Establece el número de iteraciones por defecto.
        if n_iters is None and n_epochs is None: n_iters = 200 if train_data.size <= 100000 else 600

        # Si existe un límite de longitud de entrenamiento, se divide la serie.
        if self.max_train_length is not None:
            # Separa el conjunto de datos.
            sections:int = train_data.shape[1] // self.max_train_length
            # Si hay más de una sección, se concadenan.
            if sections >= 2: train_data = np.concatenate(
                split_with_nan(
                    x=train_data,
                    sections=sections,
                    axis=1
                ),
                axis=0
            )
        
        # TODO: Detecta si hay timestamps completamente faltantes al inicio o al final.

        # TODO: Elimina series o instancias completamente NaN.

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
            self._net.parameters(),
            lr=self.learning_rage
        )

        # Listado para almacenar el loss promedio por época.
        loss_log:List[float] = []

        # Bucle de entrenamiento principal.
        while True:
            # Termina si alcanza el número máximo de épocas.
            if n_epochs is not None and self.n_epochs >= n_epochs: break

            # Acumula el loss de una época.
            cum_loss:float = 0.0
            # Cuenta el número de iteraciones por época.
            n_epoch_iters:int = 0
            # Bandera para interrumpir el entrenamiento si se alcanzan iteraciones máximas.
            interrupted:bool = False

            # Búcle por batch.
            for batch in train_loader:
                # Termina si alcanza el número máximo de iteraciones.
                if n_iters is not None and self.n_iters >= n_iters:
                    # Establece el valor como True, para indicar que se alcanzó el máximo de iteraciones.
                    interrupted = True
                    break

                # Obtiene los datos del batch.
                x:torch.Tensor = batch[0]

                # Si la secuencia es demasiado larga, se toma un sub-window aleatorio.
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    # Calcula el offset de la ventana aleatoriamente.
                    windows_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, windows_offset : windows_offset + self.max_train_length]

                # Envía los datos al dispositivo (CPU/GPU).
                x = x.to(device=self.device)

                # Calcula la longitud de la serie temporal del batch.
                ts_length:int = x.size(1)

                # TODO: Define recortes aleatorios para la pérdida contrastiva jerárquica.
                crop_length:int = np.random.randint(
                    low=2 ** (self.temporal_unit + 1),
                    high=ts_length + 1
                )
                crop_left:int = np.random.randint(low=ts_length - crop_length + 1)
                crop_right:int = crop_left + crop_length
                crop_e_left:int = np.random.randint(
                    low=crop_right,
                    high=ts_length + 1
                )
                crop_e_right:int = np.random.randint(
                    low=crop_right,
                    high=ts_length + 1
                )
                crop_offset = np.random.randint(
                    low=-crop_left,
                    high=ts_length - crop_e_right + 1,
                    size=x.size(0)
                )

                # Reinicia los gradientes del optimizador.
                optimizer.zero_grad()

                # Aplica la red a la primera vista recortada de la serie.
                out_1 = self._net(take_per_row(
                        A=x,
                        indx=crop_offset + crop_left, 
                        num_elements=crop_right - crop_e_left
                    )
                )
                out_1 = out_1[:, -crop_length]

                # Aplica la red a la segunda vista recortada
                out_2 = self._net(take_per_row(
                        A=x,
                        indx=crop_offset + crop_left, 
                        num_elements=crop_e_right - crop_left
                    )
                )
                out_2 = out_2[:, :crop_length]
                
                # Calcula la pérdida contrastiva jerárquica.
                loss:torch.Tensor = hierarchical_contrastive_loss(
                    z1=out_1,
                    z2=out_2,
                    temporal_unit=self.temporal_unit
                )

                # Backpropagation.
                loss.backward()
                optimizer.step()
                # Sincroniza parámetros.
                self.net.update_parameters(self._net)

                # Acumula loss e iteraciones.
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1

                # Callback opcional después de cada iteración.
                if self.after_iter_callback is not None:
                    # Ejecuta el callback.
                    self.after_iter_callback(self, loss.item())
            
            # Termina completamente si ha sido interrumpido por alcanzar el máximo de iteraciones.
            if interrupted: break

            # Calcula loss promedio de la época.
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)

            # Imprime loss si verbose=True.
            if verbose: print(f"Epoch |{self.n_epochs}:\tloss={cum_loss}")

            # Incrementa el contador de épocas.
            self.n_epochs += 1

            # Callback opcional después de cada época.
            if self.after_epoch_callback is not None:
                    # Ejecuta el callback.
                    self.after_epoch_callback(self, cum_loss)
        
        # Returns.
        return loss_log