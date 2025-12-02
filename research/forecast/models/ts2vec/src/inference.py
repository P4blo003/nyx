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
from typing import List
# Externos:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# Internos:
from mask import MaskGenerator
from utils import torch_pad_nan


# ==============================
# CLASES
# ==============================

class TS2VecInference:
    """
    Gstor de inferencia para TS2Vec. Se encarga de generar embeddings a partir de datos nuevos, gestionando
    ventanas deslizantes (sliding windows) y pooling de representaciones.
    """
    # ---- Default ---- #

    def __init__(
        self,
        model:nn.Module,
        device:str = 'cuda'
    ) -> None:
        """
        Inicializa el motor de inferencia.

        Args:
            model (nn.Module): Modelo de TS2Vec.
            device (str): Dispositivo de ejecución.
        """
        # Inicializa las propiedades.
        self.model:nn.Module = model
        self.device:str = device


    # ---- Métodos ---- #

    def __eval_with_pooling(
        self,
        x:torch.Tensor,
        mask_generator:MaskGenerator|None,
        encoding_window:str|int|None = None,
        slicing:slice|None = None
    ) -> torch.Tensor:
        """
        Evalúa la red TS2Vec sobre un batch de datos y aplica pooling temporal para
        obtener embeddings representativos según la ventana especificada.

        Args:
            x (torch.Tensor):
            mask_generator (MaskGenerator|None): Generador de la máscara
            encoding_window (str|int|None):
            slicing (slice|None):
        
        Returns:
            torch.Tensor:
        """
        # Se envía el tensor a la GPU si aplica y se usa mask si existe.
        out:torch.Tensor = self.model(x.to(self.device, non_blocking=True), mask_generator if mask_generator is not None else None)

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

    def encode(
        self,
        data:np.ndarray,
        mask_generator:MaskGenerator|None,
        encoding_window:str|int|None = None,
        sliding_length:int|None = None,
        sliding_padding:int = 0,
        batch_size:int|None = None,
        causal:bool = False
    ) -> np.ndarray:
        """
        Genera embeddings de los datos de entrada usando un modelo de TS2Vec.

        Args:
            data (numpy.ndarray): Serie temporal en formato (batch, time, features).
            mask_generator (MaskGenerator|None): Generador de la máscara.
            encoding_window (str|int|None): Modo de extracción de embeddings.
            causal (bool): True si se aplica codificación causal (solo usa el pasado).
            sliding_length (int|None): Longitud de ventana para procesado por sliding window.
            slinding_padding (int|None): Relleno aplicado en la ventana deslizante.
            batch_size (int|None): Tamaño del batch para slinding window.
        
        Returns:
            torch.Tensor: Embeddings generados, en CPU.
        """
        # Asegura que la entrada tiene la forma (batch, time, features).
        assert data.ndim == 3
        # Extrae el número de instancias y longitud temporal de la entrada.
        n_samples, ts_length, _ = data.shape

        # Guarda el estado original del modelo para restaurarlo después.
        oririginal_training:bool = self.model.training
        # Coloca el modelo en modo evaluación para evitar dropout, batchnorm, etc.
        self.model.eval()

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
                                    mask_generator=mask_generator,
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
                                mask_generator=mask_generator,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            # Añade los valores.
                            reprs.append(out)
                    
                    # Tras el bucle, si quedan ventanas acumuladas sin evaluar, se procesan.
                    if n_samples < batch_size and calc_buffer_length > 0:
                        out:torch.Tensor = self.__eval_with_pooling(
                            x=torch.cat(calc_buffer, dim=0),
                            mask_generator=mask_generator,
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
                        mask_generator=mask_generator,
                        encoding_window=encoding_window
                    )

                    # Si la ventana es 'full_series', elimina la dimensión temporal (1).
                    if encoding_window == 'full_series': out = out.squeeze(1)
                
                # Añade el embedding procesado del batch al buffer general.
                output.append(out)
            
            # Concatena los embeddings de todos los batches en la dimensión batch.
            output_tensor:torch.Tensor = torch.cat(output, dim=0)
        
        # Restaura el estado de entrenamiento original del modelo.
        self.model.train(oririginal_training)

        # Devuelve el embedding final en formato numpy.
        return output_tensor.numpy()