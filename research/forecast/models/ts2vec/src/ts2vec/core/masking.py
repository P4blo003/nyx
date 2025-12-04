# ==========================================================================================
# Author: Pablo González García.
# Created: 03/12/2025
# Last edited: 03/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from abc import ABC
from abc import abstractmethod
# Externos:
import torch
import numpy as np


# ==============================
# CLASES
# ==============================

class MaskGenerator(ABC):
    """
    Clase abstracta para generadores de máscaras. Define la interfaz que deben 
    implementar todas las estrategias de enmascaramiento
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int
    ) -> None:
        """
        Inicializa el generador de máscaras.

        Args:
            b (int): Número de series.
            t (int): Longitud de las series.
        """
        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t


    # ---- Métodos ---- #

    @abstractmethod
    def generate(
        self
    ) -> torch.Tensor:
        """
        Genera la máscara booleana.

        Returns:
            torch.Tensor: Máscara booleana.
        """
        pass

class BinomialMaskGenerator(MaskGenerator):
    """
    Generador de máscaras binomiales.
    """
    # ---- Default ---- #
    def __init__(
        self,
        b:int,
        t:int,
        p:float=0.5
    ) -> None:
        """
        Inicializa el generador de máscaras binomiales.

        Args:
            b (int): Número de series.
            t (int): Longitud de las series.
            p (float): Probabilidad de enmascaramiento.
        """
        # Constructor de MaskGenerator.
        super().__init__(b=b,t=t)

        # Inicializa las propiedades.
        self.p:float = p
    

    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana aleatoria según una distribución binomial
        (Bernoulli) con probabilidad `p`.

        Returns:
            torch.Tensor: Máscara booleeana de forma `(b,t)` con `False` en las posiciones
                enmascaradas y `True` en las restantes.
        """
        # Genera la máscara booleana.
        return torch.from_numpy(np.random.binomial(
            n=1,
            p=self.p,
            size=(self.b, self.t)
            )
        ).to(dtype=torch.bool)

class ContinuousMaskGenerator(MaskGenerator):
    """
    Generador de máscaras para tramos continuos.
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int,
        n:int|float = 5,
        l:int|float = 0.1
    ) -> None:
        """
        Inicializa el generador de máscaras continuas.

        Args:
            b (int): Número de series.
            t (int): Longitud de las series.
            n (int|float): Número de segmentos a enmascarar por secuencia.
                - Si `n` es `int`, se interpreta como número de bloques.
                - Si `n` es `float`, se interpreta como fracción de `t`.
            l (int|float): Longitud de cada segmento enmascarado.
                - Si `l` es `int`, se interpreta como número de timestamps a enmascarar por bloque.
                - Si `l` es `float`, se interpreta como una fracción de `t`.
        """
        # Constructor de MaskGenerator.
        super().__init__(b=b,t=t)

        # Inicializa las propiedades.
        self.n:int|float = n
        self.l:int|float = l
    

    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana que enmascara segmentos continuos. Cada secuencia del
        batch tendrá `n` segmentos enmascarados de longitud `l` cada uno.

        Returns:
            torch.Tensor: Máscara booleana de forma `(b,t)` con `False` en las posiciones
                enmascaradas y `True` en las restantes.
        """
        # Inicializa la máscara con valores completamente verdaderos.
        result:torch.Tensor = torch.full(
            size=(self.b, self.t),
            fill_value=True,
            dtype=torch.bool
        )

        # Si n es float, calcula el número de segmentos y se asegura de que al menos
        # haya un segmento de enmascaración.
        self.n = int(self.n * self.t) if isinstance(self.n, float) else self.n
        self.n = max(min(self.n, self.t // 2), 1)
        # Si l es float, calcula el número de enmascarados por segmentos y se asegura
        # de que al menos haya una enmascaración por segmento.
        self.l = int(self.l * self.t) if isinstance(self.l, float) else self.l
        self.l = max(self.l, 1)

        # Itera sobre los batches.
        for i in range(self.b):
            # Itera sobre los segmentos.
            for _ in range(self.n):
                # Elige un punto aleatorio entre [0, t-l].
                t:int = np.random.randint(low=self.t - self.l + 1)
                # Marca un bloque de longitud l como enmascarado.
                result[i, t:t+self.l] = False
        
        return result