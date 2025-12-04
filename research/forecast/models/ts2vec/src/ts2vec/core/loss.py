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
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma
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

def temporal_contrastive_loss(
    z1:torch.Tensor,
    z2:torch.Tensor
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva temporal basada en `InfoNCe` entre dos
    secuencias de emebddings.
    
    El objetivo es maximizar la similitud entre los embeddings de posiciones
    temporales correspondientes (pares positivos) y minimiar la similitud con
    el resto de posiciones (pares negativos) dentro del mismo batch/secuencia.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma
            (batch_size, timesteps, channels).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma
            (batch_size, timesteps, channels).
        
    Returns:
        torch.Tensor: Escalar con la pérdida temporal contrastiva.
    """
    # Extrae las dimensiones.
    b:int = z1.size(0)          # Tamaño del batch.
    t:int = z1.size(1)          # Número de timesteps por batch.

    # Si solo hay un timestep, no hay suficientes negativos temporales
    # que contrastar.
    if t <= 1: return z1.new_tensor(0.)

    # Concatena las dos secuencias de embeddings a lo largo de la dimensión temporal.
    z:torch.Tensor = torch.cat(tensors=[z1, z2], dim=1)

    # Calcula la matriz de similitud (producto escalar). Se calcula la similitud entre
    # cada timestep en z contra todos los demás timesteps.
    # Esto genera una matriz de similitud de 2t * 2t por cada elemento del batch.
    sim:torch.Tensor = torch.matmul(
        input=z,
        other=z.transpose(
            dim0=1,
            dim1=2,
        )
    )

    # Se elimina la diagonal principal (comparación de un timestep consigo mismo).
    # Se obtiene la parte triangular inferior, excluyendo la diagonal (-1).
    # Se corta la última columna para mantener la forma simétrica.
    logits:torch.Tensor = torch.tril(
        input=sim,
        diagonal=-1
    )[:, :, :-1]
    # Se obtiene la parte triangular superior, excluyendo la diagonal (1).
    # Se suma con la parte superior, cortando la primera columna.
    logits += torch.triu(
        input=sim,
        diagonal=1
    )[:, :, 1:]

    # Se aplica log_softmax sobre la última dimensión (comparaciones) para convertir
    # las similitudes en probabilidades logarítmicas.
    # El modelo intenta minimizar esta pérdida, lo que equivale a maximizar la probabilidad
    # del par positivo.
    logits = -F.log_softmax(
        input=logits,
        dim=-1
    )

    # Se defíne un tensor de índices temporales.
    i:torch.Tensor = torch.arange(t, device=z1.device)
    
    # Pérdida para la comparación z1 -> z1.
    loss_z1_z2 = logits[:, i, t + i -1].mean()
    # Pérdida para la comparación z2 -> z1.
    loss_z2_z1 = logits[:, t + i, i].mean()

    # Calcula el promedio de ambas direcciones para simetría y estabilidad.
    return (loss_z1_z2 + loss_z2_z1) / 2

def hierarchical_contrastive_loss(
    z1:torch.Tensor,
    z2:torch.Tensor,
    alpha:float = 0.5,
    temporal_unit:int = 0
) -> torch.Tensor:
    """
    Calcula la pérdida contrastive jerárquica entre dos secuencias de embeddings.

    Esta función aplica pérdidas contrastivas en múltiples escalas temporales, reduciendo
    progresivamente la longitud de la secuencia mediante max-pooling.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma
                (batch_size, timesteps, channels).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma
            (batch_size, timesteps, channels).
        alpha (float): Peso de la pérdida contrastiva a nivel de instancia.
        temporal_unit (int): Profundidad mínima a partir de la cual se activa la pérdida temporal.
            Esto permite que la pérdida temporal solo se aplique en resoluciones más gruesas
            (chunks grandes).
    
    Returns:
        torch.Tensor: Pérdida contrastiva promedio a lo largo de todos los niveles.
    """
    # Inicializa la pérdida acumulada en el dispositivo adecuado.
    loss:torch.Tensor = torch.tensor(0., device=z1.device)

    # ALmacena el contador de niveles jerárquicos procesados.
    cont:int = 0

    # Bucle de reducción de escala.
    while z1.size(1) > 1 and alpha != 0:
        # Se aplica périda contrastiva de instancia. Esta pérdida maximiza la similitud
        # entre la representación de los chunks.
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(
                z1=z1,
                z2=z2
            )

        # Se aplica pérdida contrastiva a nivel temporal. Esta pérdida maximiza la similitud
        # entre los embeddings correspondientes y minimiza la similitud con todos los
        # demás embeddings dentro de la secuencia.
        if cont >= temporal_unit and 1 - alpha != 0:
            loss += (1 - alpha) * temporal_contrastive_loss(
                z1=z1,
                z2=z2
            )
        
        # Actualiza la escala.
        cont += 1

        # Reduce la longitud de la secuencia a la mita. Esto genera el siguiente nivel de la
        # jerarquía.
        z1 = F.max_pool1d(
            input=z1.transpose(1, 2),
            kernel_size=2
        ).transpose(1, 2)
        z2 = F.max_pool1d(
            input=z2.transpose(1, 2),
            kernel_size=2
        ).transpose(1, 2)
    
    # Cuando la secuencia se ha reducido en una única representación (longitud 1), solo se puede
    # aplicar la pérdida a nivel de instancia, ya que no hay estructura temporal interna
    # que contrastar.
    if z1.size(1) == 1 and alpha != 0:
        loss += alpha * instance_contrastive_loss(
            z1=z1,
            z2=z2
        )
        # Incrementa el último nivel.
        cont += 1
    
    # La pérdida total se normaliza por el número total de niveles jerárquicos, en los que se
    # aplicó al menos una pérdida.
    return loss / cont