# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
# Last edited: 01/12/2025
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
import torch.nn.functional as F
from torch.optim import swa_utils
from torch.utils.data import TensorDataset, DataLoader
# Internos:
from encoder import TSEncoder
from mask import MaskMode
from loss import hierarchical_contrastive_loss
from utils import split_with_nan, take_per_row, centerize_vary_length_series, torch_pad_nan


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
        device:str = 'cuda:0',
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
            hidden_dims (int): La dimensión oculta del encoder.
            depth (int): El número de bloques residuales del encoder.
            device (int): Dispositivo de ejecución.
            learning_rate (float): El ratio de aprendizaje.
            batch_size (int): El tamaño del batch.
            max_train_length (int): La máxima longitud de la secuencia para entrenamiento. Secuencias mayores
                que este valor serán separadas en subsecuencias donde cada una tendrá una longitud menor que
                `max_train_length`.
            temporal_unit (int): La unidad mínima para realizar contraste temporal. Este parámetro ayuda a reducir el coste
                de tiempo y memoria para secuencias muy largas.
            after_iter_callback (Callable|None): Función llamada después de cada iteración.
            after_epoch_callback (Callable|None): Función llamada después de cada época.
        """
        # Inicializa las propiedades.
        self.device:str = device
        self.hidden_dims:int = hidden_dims
        self.depth:int = depth
        self.learning_rate:float = learning_rate
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

    def __eval_with_pooling(
        self,
        x:torch.Tensor,
        mask:MaskMode|None,
        encoding_window:str|int|None = None,
        slicing:slice|None = None
    ) -> torch.Tensor:
        """
        Evalúa la red TS2Vec sobre un batch de datos y aplica pooling temporal para
        obtener embeddings representativos según la ventana especificada.

        Args:
            x (torch.Tensor):
            mask (str|None):
            encoding_window (str|int|None):
            slicing (slice|None):
        
        Returns:
            torch.Tensor:
        """
        # Se envía el tensor a la GPU si aplica y se usa mask si existe.
        out:torch.Tensor = self.net(x.to(self.device, non_blocking=True), mask.value if mask is not None else None)

        # Comprueba si pooling de ventana fija.
        if isinstance(encoding_window, int):
            # Max pooling con ventana de tamaño fijo.
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)

            # Ajusta la ventana si es par.
            if encoding_window % 2 == 0: out = out[:, :-1]

            # Comprueba si se indica slicing.
            if slicing is not None: out = out[:, slicing]
        
        # Comprueba si pooling sobre toda la serie.
        elif encoding_window == "full_series":
            # Compruebas si se indica slicing.
            if slicing is not None: out = out[:, slicing]

            # Maxpooling sobre la dimensión temporal completa.
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1)
            ).transpose(1, 2)

        # Comprueba si pooling multiescala.
        elif encoding_window == "multiscale":
            # Lista donde se almacenan los embeddings a distintas escalas.
            reprs:List = []
            # Índice de escala.
            p:int = 0

            # 1 << p = 2 ** p
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,  # ventana creciente: 3,5,9,17,...
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)

                # Compruebas si se indica slicing.
                if slicing is not None: t_out = t_out[:, slicing]

                # Añade el embedding de esta escala.
                reprs.append(t_out)
                p+=1
            
            # Concatena embeddings de todas las escalas en la última dimensión
            out = torch.cat(reprs, dim=-1)
        
        # Sin pooling.
        else:
            # Comprueba si se indica slicing.
            if slicing is not None: out = out[:, slicing]
        
        # Retorna el tensor en cpu.
        return out.cpu()

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
        # Verifica que los datos de entrenamiento sean un array 3D.
        assert train_data.ndim == 3

        # Establece el número de iteraciones por defecto.
        if n_iters is None and n_epochs is None: n_iters = 200 if train_data.size <= 100000 else 600

        # Si existe un límite de longitud de entrenamiento, se divide la serie.
        if self.max_train_length is not None:
            # Separa el conjunto de datos.
            sections:int = train_data.shape[1] // self.max_train_length
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
            self._net.parameters(),
            lr=self.learning_rate
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

            # Bucle por batch.
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

                # Define recortes aleatorios para la pérdida contrastiva jerárquica.
                crop_length:int = np.random.randint(
                    low=2 ** (self.temporal_unit + 1),
                    high=ts_length + 1
                )
                crop_left:int = np.random.randint(low=ts_length - crop_length + 1)
                crop_right:int = crop_left + crop_length
                crop_e_left:int = np.random.randint(low=crop_left + 1)
                crop_e_right:int = np.random.randint(
                    low=crop_right,
                    high=ts_length + 1
                )
                crop_offset = np.random.randint(
                    low=-crop_e_left,
                    high=ts_length - crop_e_right + 1,
                    size=x.size(0)
                )

                # Reinicia los gradientes del optimizador.
                optimizer.zero_grad()

                # Aplica la red a la primera vista recortada de la serie.
                out_1 = self._net(take_per_row(
                        A=x,
                        indx=crop_offset + crop_e_left, 
                        num_elements=crop_right - crop_e_left
                    )
                )
                out_1 = out_1[:, -crop_length:]

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

    def encode(
        self,
        data:np.ndarray,
        mask:MaskMode|None = None,
        encoding_window:str|int|None = None,
        causal:bool = False,
        sliding_length = None,
        sliding_padding:int = 0,
        batch_size:int|None = None
    ):
        """
        Genera embeddings de ls datos de entrada usando TS2Vec.

        Args:
            data (numpy.ndarray): Serie temporal en formato (batch, time, features).
            mask (MaskMode): Másacara para indicar timesteps válidos.
            encoding_window (str|int|None): Modo de extracción de embeddings.
            causal (bool): True si se aplica codificación causal (solo usa el pasado).
            sliding_length (int|None): Longitud de ventana para procesado por sliding window.
            slinding_padding (int|None): Relleno aplicado en la ventana deslizante.
            batch_size (int|None): Tamaño del batch para slinding window.
        
        Returns:
            torch.Tensor: Embeddings generados, en CPU.
        """
        # Verifica que el modelo ya está entrenado o cargado antes de codificar.
        assert self.net is not None, "Por favor, entrena o carga un modelo primero."
        # Asegura que la entrada tiene la forma (batch, time, features).
        assert data.ndim == 3

        # Si no se especifica batch_size, se utiliza el mismo que durante el entrenamiento.
        if batch_size is None: batch_size = self.batch_size

        # Extrae el número de instancias y longitud temporal de la entrada.
        n_samples, ts_length, _ = data.shape

        # Guarda el estado original del modelo para restaurarlo después.
        oririginal_training:bool = self.net.training
        # Coloca el modelo en modo evaluación para evitar dropout, batchnorm, etc.
        self.net.eval()

        # Crea un dataset tensorial a partir del array de entrada.
        dataset:TensorDataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        # Crea un loader para procesar la entrada en batches.
        loader:DataLoader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )

        # Desactiva el cálculo del gradiente para acelarar y reducir memoria.
        with torch.no_grad():
            # Variable para almacenar la salida.
            output:List = []

            # Iteración por batches sobre la serie temporal completa.
            for batch in loader:
                # Obtiene el primer elemento.
                x = batch[0]

                # Si se activa slinding window, se procesarán ventanas solapadas.
                if sliding_length is not None:
                    # Variable para almacenar las representaciones.
                    reprs:List[torch.Tensor] = []

                    # Acumula ventanas antes de evaluarlas.
                    calc_buffer:List = []
                    # Longitud total acumulada.
                    calc_buffer_length:int = 0
                    
                    # Recorre toda la serie en pasos de slinding_length.
                    for i in range(0, ts_length, sliding_length):
                        # Determina la ventana deslizante (extendida por padding).
                        left:int = i - sliding_padding
                        right:int = i + sliding_length + (sliding_padding if not causal else 0)

                        # Recorta la ventana y añade padding con NaN donde sea necesario.
                        x_sliding = torch_pad_nan(
                            arr=x[:, max(left, 0):min(right, ts_length)],
                            left=-left if left < 0 else 0,
                            right=right-ts_length if right > ts_length else 0,
                            dim=1
                        )

                        # Caso especial (n_samples < batch_size). Se acumula hasta llenar un batch.
                        if n_samples < batch_size:
                            # Si al añadir el nuevo batch se supera batch_size, se evalúa lo acumulado.
                            if calc_buffer_length + n_samples > batch_size:
                                out:torch.Tensor = self.__eval_with_pooling(
                                    x=torch.cat(
                                        tensors=calc_buffer,
                                        dim=0
                                    ),
                                    mask=mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                # Divide la salida en bloques de tamaño n_samples.
                                reprs += (torch.split(out, n_samples))
                                # Reinicia los valores.
                                calc_buffer = []
                                calc_buffer_length = 0
                            
                            # Añade los valores.
                            calc_buffer.append(x_sliding)
                            calc_buffer_length += n_samples
                        
                        # Caso normal (Se evalúa directamente cada ventana deslizante).
                        else:
                            out:torch.Tensor = self.__eval_with_pooling(
                                x=x_sliding,
                                mask=mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            # Añade los valores.
                            reprs.append(out)
                    
                    # Tras el bucle, si quedan ventanas acumuladas sin evaluar, se procesan.
                    if n_samples < batch_size and calc_buffer_length > 0:
                        out:torch.Tensor = self.__eval_with_pooling(
                            x=torch.cat(calc_buffer, dim=0),
                            mask=mask,
                            slicing=slice(sliding_padding, sliding_padding+sliding_length),
                            encoding_window=encoding_window
                        )
                        # Divide la salida en bloques de tamaño n_samples.
                        reprs += (torch.split(out, n_samples))
                        # Reinicia los valores.
                        calc_buffer = []
                        calc_buffer_length = 0
                    
                    # Concatena todas las ventanas en la dimensión temporal.
                    out:torch.Tensor = torch.cat(reprs, dim=1)

                    # Si se pide 'full_series', realiza max-pooling sobre la longitud temporal.
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            input=out.transpose(1,2).contiguous(),
                            kernel_size=out.size(1)
                        ).squeeze(1)
                
                # Caso sin slinding window.
                else:
                    # Evalúa la representación base aplicando la ventana indicada.
                    out:torch.Tensor = self.__eval_with_pooling(
                        x=x,
                        mask=mask,
                        encoding_window=encoding_window
                    )

                    # Si la ventana es 'full_series', elimina la dimensión temporal (1).
                    if encoding_window == 'full_series': out = out.squeeze(1)
                
                # Añade el embedding procesado del batch al buffer general.
                output.append(out)
            
            # Concatena los embeddings de todos los batches en la dimensión batch.
            output_tensor:torch.Tensor = torch.cat(output, dim=0)
        
        # Restaura el estado de entrenamiento original del modelo.
        self.net.train(oririginal_training)

        # Devuelve el embedding final en formato numpy.
        return output_tensor.numpy()