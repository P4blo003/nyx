# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 04/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Externos:
import torch
from torch import nn
from torch import optim
# Internos:
from core.backbone import SameConv1DSequence
from core.masking import MaskGenerator, BinomialMaskGenerator, AllTrueMaskGenerator


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
        hidden_dim:int = 64,
        depth:int = 10,
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
        self.mask_gen:MaskGenerator|None = None

        # Inicializa la capa convolucional.
        self.input_fc:nn.Linear = nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_dim
        )

        # Inicializa la secuencia de bloques convolucionales.
        self.sequence:SameConv1DSequence = SameConv1DSequence(
            input_dim=self.hidden_dim,
            channels=[self.hidden_dim] * depth + [self.output_dim],
            kernel_size=kernel_size
        )

        # Inicializa un dropout para regularizar los embeddings.
        self.dropout:nn.Dropout = nn.Dropout(p=dropout)
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor,
        mask_gen:MaskGenerator|None = None
    ) -> torch.Tensor:
        """
        Ejecuta el pipeline completo del encoder.

        Args:
            x (torch.Tensor): Tensor de entrada con la forma
                (batch_size, timestamps, features).
            mask_gen (MaskGenerator): Generador de la máscara.
            
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

        # Inicializa el generador en caso de que no se haya pasado.
        if mask_gen is None:
            # Si está entrenando, la máscara empleada será Binomail por defecto.
            # en caso de no estar entrenando, no se enmascara.
            self.mask_gen = BinomialMaskGenerator(b=x.size(0), t=x.size(1)) if self.training else AllTrueMaskGenerator(b=x.size(0), t=x.size(1))
        # Si se ha pasado la máscara a emplear.
        else:
            self.mask_gen = mask_gen

        # Genera la máscara y la envía al mismo dispositivo que los datos.
        mask = self.mask_gen.generate().to(device=x.device)

        # Asegura que cualquier valor NaN sea excluido de la parte visible.
        mask &= nan_mask
        # Establece los valores elegidos para ser ocultados como 0.
        x[~mask] = 0

        # Cambia la forma de x a (batch_size, output_dim, timesteps).
        x = x.transpose(1, 2)       
        # Aplica el encoder convolucional y el dropout.
        x = self.dropout(self.sequence(x))
        # Cambia la forma de x a (batch_size, timesteps, output_dim).
        x = x.transpose(1, 2)

        return x

class SWAEncoder(nn.Module):
    """
    Contenedor que gestiona el TSEncoder activo y su versión SWA. Permite
    al trainer tratar a la instancia como un único modelo.
    """
    # ---- Default ---- #

    def __init__(
        self,
        encoder:nn.Module,
        device:str = 'cuda'
    ) -> None:
        """
        Inicializa el contenedor.

        Args:
            encoder (nn.Module): Encoder del modelo.
            device (str): Dispositivo de ejecución. Puede ser `cpu` o `cuda`.
        """
        # Constructor de nn.Module.
        super().__init__()

        # Inicializa las propiedades.
        self.device:str = device
        self.core:nn.Module = encoder.to(device=self.device)

        # Inicializa la versión promediada (SWA) que se usa para la inferencia.
        self.optimEncoder:optim.swa_utils.AveragedModel = optim.swa_utils.AveragedModel(model=self.core)
        self.optimEncoder.update_parameters(self.core)
    

    # ---- Métodos ---- #

    def forward(
        self,
        x:torch.Tensor
    ) -> torch.Tensor:
        """
        Ejectua el encoder.

        Args:
        x (torch.Tensor): Tensor de entrada con la forma
            (batch_size, timestamps, features).

        Returns:
            torch.Tensor: Tensor de salida tras aplicar el encoder con la
                forma (batch_size, timesteps, output_dim).
        """
        # Ejecuta el encoder sobre el input.
        return self.core(x)
    
    def update(
        self
    ) -> None:
        """
        Actualiza los parámetros del modelo.
        """
        # Actualiza los parámetros.
        self.optimEncoder.update_parameters(self.core)