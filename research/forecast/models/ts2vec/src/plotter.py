# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List
# Externos:
import matplotlib.pyplot as plt


# ==============================
# FUNCIONES
# ==============================

def plot_values(
    values:List,
    file_path:str
) -> None:
    """
    Genera una gráfica para los valores dados. Se interpreta que ya
    están ordenados.
    """
    # Genera la gráfica.
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(values)), values, linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    plt.grid(True)

    # Almacena la gráfica.
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close()