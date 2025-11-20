# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
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


# ==============================
# FUNCIONES
# ==============================

def instance_contrastive_loss(
    z1:torch.Tensor,
    z2:torch.Tensor
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva por instancia. Compara representaciones `` y `` para maximizar su similitud
    y minimizar la similitud con el resto de ejemplos.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma (B, T, C).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma (B, T, C).    
    
    Returns:
        torch.Tensor: Escalar con la pérdida contrastiva promedio sobre pares positivos. Valores más
            bajos implican que las representaciones de la misma instancia son más similares que las representaciones
            entre instacias distintas.
    """
    # Obtiene el tamaño del batch (número de secuencias).
    B:int = z1.size(0)

    # Si solo hay un ejemplo en el batch, no es posible formar pares positivos/negativos.
    if B == 1: return z1.new_tensor(0.)

    # Concatena las dos vistas a lo largo del batch -> forma (2B, T, C)
    # Cada secuencia aparece 2 veces, una en z1 y otra en z2.
    z:torch.Tensor = torch.concat([z1, z2], dim=0)

    # Transpone a (T, 2B, C) para calcular similitudes por timestep.
    # Cada timestep t se procesa independientemente.
    z = z.transpose(0, 1)

    # Calcula la matriz de similitud mediante producto matricial.
    sim:torch.Tensor = torch.matmul(z, z.transpose(1, 2))               # T x 2B x 2B
    
    # Selecciona todas las similitudes excepto la diagonal (i==j), separando triángulo inferior.
    # tril(diagonal=-1) deja solo elementos por debajo de la diagonal.
    # Luego se elimina la última columna para alinear tamaños.
    logits:torch.Tensor = torch.tril(sim, diagonal=-1)[:, :, :-1]       # T x 2B x (2B -1)

    # Añade el triángulo superior (por encima de la diagonal).
    # triu(diagonal=1) conserva elementos por encima de la diagonal.
    # Luego se elimina la última columna para alinear tamaños.
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    # Aplica soft-max negativo sobre la última dimensión -> convierte similitudes en pérdidas.
    # Cuanto mayor la similitud del par positivo, menor será la pérdida asociada.
    logits = -F.log_softmax(logits, dim=-1)

    # Índices 0...B-1 para extraer los pares positivos (z1_i, z2_i).
    i:torch.Tensor = torch.arange(B, device=z1.device)

    # Extrae las pérdidas de los pares positivos en ambas direcciones.
    # - Para z1 frente a z2: posición (i, B+i-1)
    # - Para z2 frente a z1: posición (B+i, i)
    # Promedia las dos luego promedia en el tiempo.
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2