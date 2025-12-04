# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 03/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List
# Internos:
import torch
import torch.nn.functional as F
from torch import nn


# ==============================
# FUNCIONES
# ==============================

def calculate_receptive_field(
    kernel_size:int,
    dilation:int
) -> int:
    """
    Calcula el tamaño efectivo del kernel considerando la dilatación.

    Args:
        kernel_size (int): Tamaño del filtro de la convolución.
        dilation (int): Dilatación de la convolución.
    
    Returns:
        int: Campo receptivo del kernel.
    """
    # Calcula el campo receptivo.
    return (kernel_size - 1) * dilation + 1


# ==============================
# CLASES
# ==============================

class SameConv1D(nn.Module):
    """
    Capa de convolución 1D con padding `same` que garantiza que la salida tenga la
    misma longitud temporal que la entrada, independientemente del tamaño del
    kernel o la dilatación.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        kernel_size:int,
        dilation:int = 1,
        groups:int = 1,
        bias:bool = True
    ) -> None:
        """
        Inicializa la capa convolucional 1D con padding `same`.

        Args:
            input_dim (int): Número de canales de entrada de la convolución. Por ejemplo,
                en el caso de una serie temporal con 3 features, `input_dim` = 3.
            output_dim (int): Número de canales de salida de la convolución. Suele
                coincidir con `input_dim` para mantener dimensionalidad de embedding, 
                pero puede ser distinto si se desea cambiar el tamaño de los embeddings.
            kernel_size (int): Tamaño del filtro de la convolución. Controla cuántos
                pasos temporales se consideran en cada operación convolucional.
            dilation (int): Dilatación de la convolución, es decir, el espaciado entre
                los elementos del kernel. Valores >1 permiten capturar patrones a más
                largo plazo sin aumentar el kernel.
            groups (int): Permite hacer convoluciones agrupadas. Si `group`=1, se trata
                de una convolución estándar y si `groups`=`input_dim` hace depthwise
                convolution.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Calcula el campo receptivo efectivo (kernel dilatado).
        self.receptive_field:int = calculate_receptive_field(
            kernel_size=kernel_size,
            dilation=dilation
        )
        # Define la capa convolucional 1D con padding.
        self.conv:nn.Conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=self.receptive_field // 2,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # Si el receptive_field es par, la salida se 1 más larga y hay que recortar.
        self.remove_last:bool = self.receptive_field % 2 == 0
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta la capa convolucional.

        Args:
            x (torch.Tensor): Tensor de entrada con la forma
                (batch_size, output_dim, timesteps).
            
        Returns:
            torch.Tensor: Tensor de salida tras aplicar la capa convolucional con
                las mismas dimensiones que la entrada.
        """
        # Aplica la convolución.
        out:torch.Tensor = self.conv(x)

        # Ajusta la convolución si el receptive_field es par.
        if self.remove_last: out[:, :, :-1]

        return out

class SameConv1DBlock(nn.Module):
    """
    Bloque convolucional para series temporales que combina varias capas convolucionales
    `SameConv1D`. Además aplica funciones de activación y una función residual para
    mejorar la propagación del gradiente.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        kernel_size:int,
        dilation:int,
        final:bool = False
    ) -> None:
        """
        Inicializa el bloque convolucional.

        Args:
            input_dim (int): Número de canales de entrada del bloque. Por ejemplo, para
                una serie temporal con  3 features, `input_dim`=3.
            output_dim (int): Número de canales de salida de la convolución. Suele
                coincidir con `input_dim` para mantener dimensionalidad de embedding, 
                pero puede ser distinto si se desea cambiar el tamaño de los embeddings.
            kernel_size (int): Tamaño del filtro de la convolución. Controla cuántos
                pasos temporales se consideran en cada operación convolucional.
            dilation (int): Dilatación de la convolución, es decir, el espaciado entre
                los elementos del kernel. Valores >1 permiten capturar patrones a más
                largo plazo sin aumentar el kernel.
            final (bool): Indica si este bloque es el final del bloque.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa una capa de proyección (convolución 1D) para alinear dimensiones
        # en la conexión residual. Se aplica si los canales de entrada y salida no
        # coinciden, o si es el bloque final.
        self.projector:nn.Conv1d|None = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=1
        ) if input_dim != output_dim or final else None
        # Crea la primera capa convolucional.
        self.conv1:SameConv1D = SameConv1D(
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            dilation=dilation
        )
        # Crea la segunda capa convolucional.
        self.conv2:SameConv1D = SameConv1D(
            input_dim=output_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            dilation=dilation
        )
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta el bloque convolucional.

        Args:
            x (torch.Tensor): Tensor de entrada con la forma
                (batch_size, output_dim, timesteps).
            
        Returns:
            torch.Tensor: Tensor de salida tras aplicar el bloque convolucional con
                las mismas dimensiones que la entrada.
        """
        # Inicializa la conexión residual. Se usa para la entrada directa o
        # proyectada para ajustar canales.
        residual:torch.Tensor = x if self.projector is None else self.projector(x)

        # Aplica una función de activación antes de la primera capa.
        x = F.gelu(x)
        x = self.conv1(x)
        # Aplica una función de activación antes de la segunda capa.
        x = F.gelu(x)
        x = self.conv2(x)

        # Suma la entrada/residual para la conexión residual.
        return x + residual

class SameConv1DSequence(nn.Module):
    """
    Secuencia de bloques convolucionales de `SameConv1D`. Extrae representaciones
    jerárquicas y multiescala de la señal aumentando progresivamente el campo receptivo
    sin incrementar el coste computacional de forma significativa.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dim:int,
        channels:List[int],
        kernel_size:int
    ) -> None:
        """
        Inicializa la secuencia de bloques.

        Args:
            input_dim (int): Número de canales de entrada del bloque. Por ejemplo, para
                una serie temporal con  3 features, `input_dim`=3.
            channels (List[int]): Listado que contiene dimensiones de salida. Se generan
                tantos bloques convolucionales como tamaño tenga `channels`.
            kernel_size (int): Tamaño del filtro de la convolución. Controla cuántos
                pasos temporales se consideran en cada operación convolucional.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa la red de bloques.
        self.sequence:nn.Sequential = nn.Sequential(*[
            SameConv1DBlock(
                input_dim=channels[i-1] if i > 0 else input_dim,
                output_dim=channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels) - 1)
            ) for i in range(len(channels))
        ])
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta la secuencia de bloques convolucionales.

        Args:
            x (torch.Tensor): Tensor de entrada con la forma
                (batch_size, output_dim, timesteps).
            
        Returns:
            torch.Tensor: Tensor de salida tras aplicar la sequencia de bloques
            convolucionales con las mismas dimensiones que la entrada.
        """
        # Aplica la secuencia.
        return self.sequence(x)