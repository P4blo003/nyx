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
from typing import List
# Externos:
import torch
from torch import nn
# Internos:
from conv import SameConv1dBlock
from mask import MaskMode
from mask import generate_binomial_mask, generate_continuous_mask


# ==============================
# CLASES
# ==============================

class SameConv1dEncoder(nn.Module):
    """
    Encoder convolucional para series temporales basado en convoluciones 1D con padding.
    Su objetivo es extraer representaciones jerárquicas y multiescala de la señal aumentando
    progresivamente el campo receptivo sin incrementar el coste computacional de forma
    significativa. 
    """
    # ---- Default ---- #

    def __init__(
        self,
        in_channels:int,
        channels:List[int],
        kernel_size:int
    ) -> None:
        """
        Inicializa el encoder.

        Args:
            in_channels (int): Número de canales de entrada de la convolución. Por ejemplo, en el caso de una
                serie temporal multivariada con 3 features, `in_channels`=3.
            channels (List[int]): Listado que contiene los números de canales de salida.
            kernel_size (int): Tamaño del filtro de la convolución. Controla cuántos pasos temporales se consideran
                en cada operación convolucional. Afecta directamente al `receptive_field`, es decir, el rango temporal
                de observación de cada salida.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa la red de bloques.
        self.net:nn.Sequential = nn.Sequential(*[
            SameConv1dBlock(
                in_channels=channels[i-1] if i > 0 else in_channels,
                out_channels=channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels) - 1)
            ) for i in range (len(channels))
        ])

    
    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta la red de convoluciones.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch, channels, sequence_length).
        
        Returns:
            torch.Tensor: Tensor de salida con la misma longitud temporal que la entrada.
        """
        # Aplica la red de capas.
        return self.net(x)

class TSEncoder(nn.Module):
    """
    Encoder de series temporales que transforma entradas multivariadas en embeddings temporales.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dims:int,
        output_dims:int,
        hidden_dims:int = 64,
        depth:int = 10,
        mask_mode:MaskMode|None = MaskMode.BINOMIAL,
        kernel_size:int = 3,
        dropout_p:float = 0.1
    ) -> None:
        """
        Inicializa el encoder de series temporales.

        Args:
            input_dims (int): Dimensión de entrada (features por timestep).
            output_dims (int): Dimensión del embedding de salida por timestep.
            hidden_dims (int): Número de neuronas ocultas.
            depth (int): Profundidad del encoder convolucional.
            mask_mode (MaskMode): Modo de enmascaramiento.
            kernel_size (int): Tamaño del kernel de las capas convolucionales.
            dropout_p (float): Probabilidad de que cada elemento sea anulado (puesto a "cero")
                durante el entrenamiento.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa las propiedades.
        self.input_dims:int = input_dims
        self.output_dims:int = output_dims
        self.hidden_dims:int = hidden_dims
        self.mask_mode:MaskMode|None = mask_mode

        # Inicializa una capa lineal para proyectar las características de entrada.
        self.input_fc:nn.Linear = nn.Linear(
            in_features=input_dims,
            out_features=hidden_dims
        )

        # Inicializa el encoder convolucional.
        self.feature_extractor:SameConv1dEncoder = SameConv1dEncoder(
            in_channels=hidden_dims,
            channels=[hidden_dims]*depth + [output_dims],
            kernel_size=kernel_size
        )

        # Inicializa un dropout para regularizar los embeddings.
        self.repr_dropout:nn.Dropout = nn.Dropout(p=dropout_p)
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor,
        mask_mode:MaskMode|None = None
    ) -> torch.Tensor:
        """
        Ejecuta el encoder.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch, channels, sequence_length).
        
        Returns:
            Embeddings de salida con forma (B, T, output_dims), donde cada timestep ha sido procesado.
        """
        # Genera una máscara para valores NaN: marca los timesteps que no tienen NaNs en ninguna característica.
        nan_mask:torch.Tensor = ~x.isnan().any(dim=-1)

        # Para los timesteps con NaN, se pone la entrada a 0 antes del procesamiento.
        x[~nan_mask] = 0

        # Se realiza la proyección lineal inicial.
        x = self.input_fc(x)

        # Decide que máscara usar si no se ha proporcionado.
        if mask_mode is None:
            # Comprueba si está en entrenamiento y se haya pasado máscara en el constructor.
            if self.training and self.mask_mode is not None:
                # Establece el valor asignado.
                mask_mode = self.mask_mode
            # En caso de que no se este entrenando o no se haya pasado máscara en el constructor.
            else:
                # Establece máscara por defecto.
                mask_mode = MaskMode.ALL_TRUE
        
        # Variable para almacenar la máscara.
        mask:torch.Tensor|None = None
        # Match case para máscara.
        match(mask_mode):
            # Caso BINOMIAL.
            case MaskMode.BINOMIAL:
                # Enmascara valores aleatorios.
                mask = generate_binomial_mask(
                    B=x.size(0),
                    T=x.size(1)  
                ).to(device=x.device)

            # Caso CONTINUOUS.
            case MaskMode.CONTINUOUS:
                # Enmascara valores continuos.
                mask = generate_continuous_mask(
                    B=x.size(0),
                    T=x.size(1)  
                ).to(device=x.device)

            # Caso ALL_FALSE.
            case MaskMode.ALL_FALSE:
                # Enmascara todo los valores.
                mask = x.new_full(
                    size=(
                        x.size(0),
                        x.size(1)
                    ), 
                    fill_value=False, 
                    dtype=torch.bool
                )

            # Caso MASK_LAST.
            case MaskMode.MASK_LAST:
                # No enmascara ningún valor.
                mask = x.new_full(
                    size=(
                        x.size(0),
                        x.size(1)
                    ), 
                    fill_value=True, 
                    dtype=torch.bool
                )
                # Solo enmascara el último.
                mask[:, -1] = False

            # Caso ALL_TRUE.
            case MaskMode.ALL_TRUE:
                # No enmascara ningún valor.
                mask = x.new_full(
                    size=(
                        x.size(0),
                        x.size(1)
                    ), 
                    fill_value=True, 
                    dtype=torch.bool
                )
        
        # Comprueba que se haya creado la máscara.
        if mask is not None: 
            # Se asegura de que respete donde hay NaNs.
            mask &= nan_mask
            # Aplica la máscara.
            x[~mask] = 0

        # Prepara la convolución: Intercambia dimensiones para que Conv1d trabaje sobre el eje temporal.
        x = x.transpose(1, 2)       # Cambia a: (B, output_dims, T)
        # Aplica el encoder convolucional y el dropout.
        x = self.repr_dropout(self.feature_extractor(x))
        # Reordena dimensiones para volver a (B, T, output_dims).
        x = x.transpose(1, 2)

        # Returns.
        return x