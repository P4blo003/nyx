# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 03/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Externos:
import torch
from torch import nn
# Internos:
from .backbone import SameConv1DSequence


# ==============================
# CLASES
# ==============================

class TSEncoder(nn.Module):
    """
    Encoder de series temporales. Transforma entradas en embeddings
    temporales.
    """
    # ---- Default ---- #

    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        hidden_dim:int,
        depth:int,
        kernel_size:int = 3,
        dropout:float = 0.1
    ) -> None:
        """
        Inicializa el encoder de series temporales.

        Args:
            input_dim (int): Número de canales de la entrada. Por ejemplo,
                en el caso de una serie temporal con 3 features, `input_dim` = 3.
            output_dim (int): Tamaño del embedding generado.
            hidden_dim (int): Número de neuronas ocultas.
            depth (int): Profundidad del ...
            kernel_size (int): Tamaño del kernel de las capas convolucionales.
            dropout (float): Probabilidad de que cada elemento sea anulado durante
                el entrenamiento.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa las propiedades.
        self.input_dim:int = input_dim
        self.output_dim:int = output_dim
        self.hidden_dim:int = hidden_dim

        # Inicializa la capa convolucional.
        self.input_fc:nn.Linear = nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_dim
        )

        # Inicializa la secuencia de bloques convolucionales.
        self.sequence:SameConv1DSequence = SameConv1DSequence(
            input_dim=self.input_dim,
            channels=[self.hidden_dim] * depth + [self.output_dim],
            kernel_size=kernel_size
        )

        # Inicializa un dropout para regularizar los embeddings.
        self.dropout:nn.Dropout = nn.Dropout(p=dropout)
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejecuta el pipeline completo del encoder.

        Args:
            x (torch.Tensor): Tensor de entrada con la forma
                (batch_size, timestamps, features).
            
        Returns:
            torch.Tensor: Tensor de salida tras aplicar el encoder con la
                forma (batch_size, timesteps, output_dim).
        """
        # Genera una máscara para valores NaN. Marca los timesteps que no tienen
        # NaNs en ninguna característica.
        # nan_mask tiene la misma forma que x, donde False indica que
        # tiene algún valor NaN y True en caso contrario.
        nan_mask:torch.Tensor = ~x.isnan().any(dim=-1)      # -1 = Última dimensión.
        # Establece los valores de x que tengan NaN como 0.
        x[~nan_mask] = 0

        # Realiza la proyección inicial.
        x = self.input_fc(x)

        # Cambia la forma de x a (batch_size, output_dim, timesteps).
        x = x.transpose(1, 2)       
        # Aplica el encoder convolucional y el dropout.
        x = self.dropout(self.sequence(x))
        # Cambia la forma de x a (batch_size, timesteps, output_dim).
        x = x.transpose(1, 2)

        return x