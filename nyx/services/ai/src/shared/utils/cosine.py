# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2026
# Last edited: 22/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
import numpy as np


# ==============================
# FUNCTIONS
# ==============================

def cosine_similarity(a, b):
    """
    """

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Verificación de normalización para debug
    for name, n in [("A", norm_a), ("B", norm_b)]:
        print(f"Norma: {n}")
        if not np.isclose(n, 1.0, atol=1e-3):
            print(f"⚠️ Alerta: El vector {name} NO está normalizado (Norma: {n:.6f}).")
    
    # Cálculo de similitud
    return np.dot(a, b) / (norm_a * norm_b)