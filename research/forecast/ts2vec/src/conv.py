# ==========================================================================================
# Author: Pablo González García.
# Created: 19/11/2025
# Last edited: 20/11/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Externos:
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
    # Retorna el campo receptivo.
    return (kernel_size - 1) * dilation + 1


# ==============================
# CLASES
# ==============================

class SameConv1d(nn.Module):
    """
    Capa de convolución 1D con paadding 'same' quee garantiza que la salida tenga la misma longitud
    temporal que la entrada, independientemente del tamaño del kernel o la dilatación.

    Esta capa es útil para redes de series temporales donde la longitud de la secuencia debe mantenerse
    a lo largo de múltiples convolucionees, como en TS2Vec.
    """
    # ---- Default ---- #

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        dilation:int = 1,
        groups:int = 1,
        bias:bool = True
    ) -> None:
        """
        Inicializa la capa convolicional 1D.

        Args:
            in_channels (int): Número de canales de entrada de la convolución. Por ejemplo, en el caso de una
                serie temporal multivariada con 3 features, `in_channels`=3.
            out_channels (int): Número de canales de salida de la convolución. Suele coincidir con `in_channels` 
                para mantener dimensionalidad de embedding, pero puede ser distinto si se desea cambiar el tamaño
                de los embeddings.
            kernerl_size (int): Tamaño del filtro de la convolución. Controla cuántos pasos temporales se consideran
                en cada operación convolucional. Afecta directamente al `receptive_field`, es decir, el rango temporal
                de observación de cada salida.
            dilation (int): Dilatación de la convolución, es decir, el espaciado entre los elementos del kernel.
                Valores > 1 permiten capturar patrones a más largo plazo sin aumentar el kernel. El `receptive_field`
                se multiplica por este valor para calcular el padding correcto.
            groups (int): Permite hacer convoluciones agrupadas. Si `groups` = 1, se trata de una convolución estándar y si
                `groups` = `in_channels` hace depthwise convolution, útil para reducir parámetros en redes profundas.
            bias (bool): Indica si la capa convolucional incluye el término de sesgo aprendido. Para cada salida se añade
                un valor escalar después de aplicar la convolución con los pesos del kernel.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Calcula el campo receptivo efectivo: kernel dilatado.
        self.receptive_field:int = calculate_receptive_field(
            kernel_size=kernel_size,
            dilation=dilation
        )

        # Calcula el padding necesario para que la salida tenga la misma longitud que la entrada.
        padding:int = self.receptive_field // 2

        # Define la capa convolucional 1D con padding.
        self.conv:nn.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Si el receptive_field es par, la salida es +1 más larga y hay que recortar.
        self.remove:bool = self.receptive_field % 2 == 0
    

    # ---- Methods ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta la convolución.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch, channels, sequence_length).
        
        Returns:
            torch.Tensor: Tensor de salida con la misma longitud temporal que la entrada.
        """
        # Aplica la convolución.
        out:torch.Tensor = self.conv(x)

        # Ajusta la longitud si el receptive_field es par.
        if self.remove: out = out[:, :, :-1]

        # Retorna el tensor.
        return out

class SameConv1dBlock(nn.Module):
    """
    Bloque convolucional para series temporales que combina varias capas convolucionales 1D con padding 'same'
    funciones de activación, una conexión residual para mejorar la propagación del gradiente y una proyección lineal
    opcional para ajustar el número de canales si es necesario.
    """
    # ---- Default ---- #

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        dilation:int,
        final:bool = False
    ) -> None:
        """
        Inicializa el bloque convolucional.

        Args:
            in_channels (int): Número de canales de entrada de la convolución. Por ejemplo, en el caso de una
                serie temporal multivariada con 3 features, `in_channels`=3.
            out_channels (int): Número de canales de salida de la convolución. Suele coincidir con `in_channels` 
                para mantener dimensionalidad de embedding, pero puede ser distinto si se desea cambiar el tamaño
                de los embeddings.
            kernerl_size (int): Tamaño del filtro de la convolución. Controla cuántos pasos temporales se consideran
                en cada operación convolucional. Afecta directamente al `receptive_field`, es decir, el rango temporal
                de observación de cada salida.
            dilation (int): Dilatación de la convolución, es decir, el espaciado entre los elementos del kernel.
                Valores > 1 permiten capturar patrones a más largo plazo sin aumentar el kernel. El `receptive_field`
                se multiplica por este valor para calcular el padding correcto.
            final (bool): Indica si este bloque es el final de la red.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Crea la capa de proyección (convolución 1x1) para alinear dimensiones en la conexión residual.
        # Se aplica si los canales de entrada y salida no coinciden, o si es el bloque final.
        self.projector:nn.Conv1d|None = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        ) if in_channels != out_channels or final else None

        # Crea la primera capa convolucional.
        self.conv1:SameConv1d = SameConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        # Crea la segunda capa convolucional.
        self.conv2:SameConv1d = SameConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )


    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta el bloque de convoluciones.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch, channels, sequence_length).
        
        Returns:
            torch.Tensor: Tensor de salida con la misma longitud temporal que la entrada.
        """
        # Conexión residual: se usa la entrada directa o la proyectada para ajustar canales.
        residual:torch.Tensor = x if self.projector is None else self.projector(x)

        # Aplica la función de activación antes de la primera capa.
        x = F.gelu(x)
        # Aplica la primera capa convolucional.
        x = self.conv1(x)
        # Aplica la función de activación antes de la segunda capa.
        x = F.gelu(x)
        # Aplica la segundaa capa convolucional.
        x = self.conv2(x)
        
        #  Suma la entrada/residual para la conexión residual.
        return x + residual

