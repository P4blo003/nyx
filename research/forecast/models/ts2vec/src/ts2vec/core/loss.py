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
import torch.nn.functional as F


# ==============================
# FUNCIONES
# ==============================

def instance_contrastive_loss(
        z1:torch.Tensor,
        z2:torch.Tensor
) -> torch.Tensor:
    """
    Calcula la loss contrastiva por instancia. Compara representacionees `z1` y `z2` para
    maximizar la similitud entre ellos y minimizarla con el resto de ejemplos.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma
            (batch_size, timesteps, channels).
        z1 (torch.Tensor): Segundo conjunto de representaciones, con forma
            (batch_size, timesteps, channels).
        
    Returns:
        torch.Tensor: Escalar con la pérdida contrastiva promedio sobre pares positivos.
            Valores más bajos indican que las representaciones de la misma instancia son
            más similares que las representaciones entre instancias distintas.
    """
    # Obtiene el tamaño del batch (Se asume que z1 y z2 tienen el mismo).
    b:int = z1.size(0)
    # En el caso de que solo haya un elemento en el batch, no es posible
    # formar pares negativos/positivos.
    if b == 1: return z1.new_tensor(data=0.)

    # Concatena las dos vistas a lo largo del batch (2 * batch_size, timesteps, channels).
    # Cada secuencia aparece 2 veces, una en z1 y otra en z2.
    z:torch.Tensor = torch.concat([z1, z2], dim=0)

    # Transpone el tensor a (timesteps, 2 * batch_size, channels) para calcular
    # similitudes por timesteps. Cada timestep se procesa independientemente.
    z = z.transpose(0, 1)

    # Calcula la matriz de similitud mediante producto matricial.
    sim:torch.Tensor = torch.matmul(    
        input=z,                    # (timesteps, 2 * batch_size, channels)
        other=z.transpose(1, 2)     # (timesteps, channels, 2 * batch_size)
    )

    # Selecciona todas las similitudes excepto la diagonal (i==j), separando
    # el triángulo inferior.
    logits:torch.Tensor = torch.tril(
        input=sim,
        diagonal=-1
    )[:, :, :-1]
    # Añade el triángulo superior (por encima de la diagonal). Luego
    # se elimina la última columna para alinear tamaños.
    logits += torch.triu(
        input=sim,
        diagonal=1
    )[:, :, 1:]

    # Aplica softmax negativo sobre la última dimensión. Convierte similitudes en
    # pérdidas. Cuanto mayor sea la similitud del par positivo, menor será la
    # pérdidas asociada.
    logits = -F.log_softmax(logits, dim=-1)

    # Índices 0...batch_size-1 para extraer pares positivos (z1_i, z2_i).
    i:torch.Tensor = torch.arange(b, device=z1.device)

    # Extrae las pérdidas de los pares positivos en ambas direcciones.
    # - Para z1 frente a z2: posición (i, batch_size+i-1).
    # - Para z2 frente a z1: posición (batch_size+i, i).
    return (logits[:, i, b + i -1].mean() + logits[:, b + i, i].mean()) / 2