# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
# Last edited: 02/12/2025
#
# Algunas partes del código han sido tomadas y adaptadas del repositorio oficial
# de TS2Vec (https://github.com/zhihanyue/ts2vec).
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
    Clase abstracta base para generadores de máscaras.

    Define la interfaz que deben implementar todas las estrategias de enmascaramiento, lo que
    permite desacoplar la lógica de generación de máscaras del modelo que las usa.
    """
    # ---- Métodos ---- #

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana

        Returns:
            torch.Tensor: Mácara generada.
        """
        pass

class BinomialMaskGenerator(MaskGenerator):
    """
    Generador de mácara binomial (Bernoulli).
    
    Genera una máscara booleana aleatoria según una distribución binomial (Bernoulli)
    con probabilidad `p`.
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int,
        p:float = 0.5
    ) -> None:
        """
        Inicializa el generador binomial.

        Args:
            b (int): Número de secuencias / ejemplos en el batch (batch size).
            t (int): Longitud temporal de cada secuencia (número de timestamps).
            p (float): Probabilidad de que la máscara sea True en cada posición.
        """
        # Constructor de MaskGenerator.
        super().__init__()

        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t
        self.p:float = p


    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana aleatoria según una distribución binomial (Bernoulli)
        con probabilidad `p`.
        
        Returns:
            torch.Tensor: Tensor booleano de forma `(B, T)` con valores True o False.
                Se emplea para enmascarar pasos temporales en modelos de series temporales.
        """
        # Genera la mácara binomial.
        return torch.from_numpy(np.random.binomial(
            n=1,
            p=self.p,
            size=(self.b, self.t)
        )).to(dtype=torch.bool)

class ContinuousMaskGenerator(MaskGenerator):
    """
    Generador de mácara para tramos continuos.

    Enmascara segmentos contiguos de tiempo, útil para obligar al model a aprender
    dependencias a largo plazo.
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
        Inicializa el generador binomial.

        Args:
            b (int): Número de secuencias / ejemplos en el batch (batch size).
            t (int): Longitud temporal de cada secuencia (número de timestamps).
            n (int|float): Número de segmentos a enmascarar por secuencia.
                - Si es un `int`, se interpreta como número de bloques.
                - Si es un `float`, se interpreta como una fracción de `t`.
            l (int|float): Longitud de cada segmento enmascarado.
                - Si es un `int`, se interpreta como número de timestamps a enmascarar por bloque.
                - Si es un `float`, se interpreta como una fracción de `t`.
        """
        # Constructor de MaskGenerator.
        super().__init__()

        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t
        self.n:int|float = n
        self.l:int|float = l
    

    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana que enmascara segmentos continuos. Cada secuencia del
        batch tendrá `n` segmentos enmascarados de longitud `l` cada uno.

        Returns:
            torch.Tensor: Máscara booleana de forma `(B, T)` con `False` en las posiciones enmascaradas
            (segmentos continuos generados aleatoriamente) y `True` en las restantes.
        """
        # Iniciailiza la máscara completamente verdadera (sin enmascaramiento).
        res:torch.Tensor = torch.full(
            size=(self.b, self.t),
            fill_value=True,
            dtype=torch.bool
        )

        # Si n es un float, calcula el número de segmentos.
        self.n = int(self.n * self.t) if isinstance(self.n, float) else self.n
        # Se asegura de que al menos haya un segmento y como mucho t//2.
        self.n = max(min(self.n, self.t // 2), 1)

        # Si l es float, calcula el número de timestamps por segmento.
        self.l = int(self.l * self.t) if isinstance(self.l, float) else self.l
        # Se asegura de que al menos haya 1 timestamp falso.
        self.l = max(self.l, 1)

        # Itera sobre las secuencias del batch.
        for i in range(self.b):
            for _ in range(self.n):
                # Elige aleatoriamiente un punto del inicio entre [0, t-l].
                t = np.random.randint(int(self.t - self.l + 1))
                # Marca un bloque de longitud l como enmascarado.
                res[i, t:t+self.l] = False
        
        return res

class AllFalseMaskGenerator(MaskGenerator):
    """
    Generador de mácara completa. Enmascara todas las posiciones.
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int
    ) -> None:
        """
        Inicializa el generador de todo falsos.

        Args:
            b (int): Número de secuencias / ejemplos en el batch (batch size).
            t (int): Longitud temporal de cada secuencia (número de timestamps).
        """
        # Constructor de MaskGenerator.
        super().__init__()

        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t


    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana que enmascara todos los valores.

        Returns:
            torch.Tensor: Máscara booleana de forma `(B, T)` con `False` en todas las posiciones.
        """
        # Retorna la máscara generada.
        return torch.full(
            size=(self.b, self.t),
            fill_value=False,
            dtype=torch.bool
        )

class AllTrueMaskGenerator(MaskGenerator):
    """
    Generador de mácara vacía. No enmacara ninguna posición.
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int
    ) -> None:
        """
        Inicializa el generador de todo falsos.

        Args:
            b (int): Número de secuencias / ejemplos en el batch (batch size).
            t (int): Longitud temporal de cada secuencia (número de timestamps).
        """
        # Constructor de MaskGenerator.
        super().__init__()

        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t


    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana no enmascara ningún valor.

        Returns:
            torch.Tensor: Máscara booleana de forma `(B, T)` con `True` en todas las posiciones.
        """
        # Retorna la máscara generada.
        return torch.full(
            size=(self.b, self.t),
            fill_value=True,
            dtype=torch.bool
        )

class LastMaskGenerator(MaskGenerator):
    """
    Generador de mácara. Enmascara el último elemento de la máscara.
    """
    # ---- Default ---- #

    def __init__(
        self,
        b:int,
        t:int
    ) -> None:
        """
        Inicializa el generador de todo falsos.

        Args:
            b (int): Número de secuencias / ejemplos en el batch (batch size).
            t (int): Longitud temporal de cada secuencia (número de timestamps).
        """
        # Constructor de MaskGenerator.
        super().__init__()

        # Inicializa las propiedades.
        self.b:int = b
        self.t:int = t


    # ---- Métodos ---- #

    def generate(self) -> torch.Tensor:
        """
        Genera una máscara booleana que enmascara todos los valores.

        Returns:
            torch.Tensor: Máscara booleana de forma `(B, T)` con `False` en todas las posiciones.
        """
        # Genera la mácara sin enmascaramiento.
        mask:torch.Tensor = AllTrueMaskGenerator(b=self.b, t=self.t).generate()
        # Establece el último elemento como enmascarado.
        mask[:, -1] = False
        # Retorna la máscara.
        return mask