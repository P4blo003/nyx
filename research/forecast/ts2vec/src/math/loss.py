# ==========================================================================================
# Author: Pablo González García.
# Created: 20/11/2025
# Last edited: 24/11/2025
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
    Calcula la pérdida contrastiva por instancia. Compara representaciones `z1` y `z2` para maximizar su similitud
    y minimizar la similitud con el resto de ejemplos.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma (B, T, C).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma (B, T, C).    
    
    Returns:
        torch.Tensor: Escalar con la pérdida contrastiva promedio sobre pares positivos. Valores más
            bajos implican que las representaciones de la misma instancia son más similares que las representaciones
            entre instancias distintas.
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

def temporal_contrastive_loss(
    z1:torch.Tensor,
    z2:torch.Tensor
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva temporal entre dos secuencias de embeddings.
    Esta pérdida mide la similitud entre posiciones temporales correspondientes
    de dos vistas z1 y z2, penalizando que timestamps incorrectos tengan mayor
    que los correctos.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma (B, T, C).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma (B, T, C).
    
    Returns:
        torch.Tensor: Escalar con la pérdida temporal contrastiva.
    """
    # Obtiene el batch size (B) y longitud temporal (T) de las secuencias.
    B:int = z1.size(0)
    T:int = z1.size(1)

    # Si solo hay un timestamp, no hay estructura temporal que comparar.
    if T == 1: return z1.new_tensor(0.)

    # Concatena en dimensión temporal ambas vistas.
    z = torch.cat(
        tensors=[z1, z2],
        dim=1
    )

    # Matriz de similitud completa entre todas las posiciones temporales.
    sim:torch.Tensor = torch.matmul(
        input=z,
        other=z.transpose(
            dim0=1,
            dim1=2
        )
    )

    # Se elimina las diagonales porque representan autocomparación.
    # Parte triangular inferior, excluyendo diagonal.
    logits:torch.Tensor = torch.tril(
        input=sim,
        diagonal=-1
    )[:, :, :-1]
    # Parte triangular superior desplazada.
    logits += torch.triu(
        input=sim,
        diagonal=1
    )[:, :, 1:]

    # Se aplica softmax negativo para convertir similitudes en pérdidas.
    logits = -F.log_softmax(
        input=logits,
        dim=-1
    )

    # t = [0, 1, ..., T-1] para indexar pares temporales correspondientes.
    t:torch.Tensor = torch.arange(
        T,
        device=z1.device
    )

    # Se promedian ambas direcciones para obtener simetría y estabilidad.
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(
    z1:torch.Tensor,
    z2:torch.Tensor,
    alpha:float = 0.5,
    temporal_unit:int = 0
) -> torch.Tensor:
    """
    Calcula la pérdida contrastiva jerárquica entre dos secuencias de embeddings.
    Este procedimiento aplica pérdidas contrastivas en múltiples escalas temporales
    reduciendo progresivamente la resolución mediante max-pooling.

    Args:
        z1 (torch.Tensor): Primer conjunto de representaciones, con forma (B, T, C).
        z2 (torch.Tensor): Segundo conjunto de representaciones, con forma (B, T, C).
        alpha (float): Peso de la pérdida contrastiva a nivel de instancia.
            El peso de la pérdida a nivel temporal es (1- alpha).
        temporal_unit (int): Profundidad mínima (número de escalas) a partir de la cual
            activar la pérdida temporal.
    
    Returns:
        torch.Tensor: Pérdida contrastiva promedio a lo largo de todos los niveles jerárquicos.
    """
    # Inicializa la pérdida acumulada en el dispositivo adecuado.
    loss = torch.tensor(0., device=z1.device)

    # Contador del número de niveles jerárquicos aplicados.
    d:int = 0

    # Bucle que continua mientras las secuencias tengan longitud > 1.
    while z1.size(1) > 1:
        
        # Calcula la pérdida contrastiva a nivel de instancia.
        if alpha != 0:
            # Calcula la pérdida de instancia.
            loss+= alpha * instance_contrastive_loss(
                z1=z1,
                z2=z2
            )
        
        # Cuando la escala actual alcanza temporal_unit.
        if d >= temporal_unit:
            if 1 - alpha > 0:
                # Calcula la pérdida temporal.
                loss += (1 - alpha) * temporal_contrastive_loss(
                    z1=z1,
                    z2=z2
                )
        
        # Incrementa el número de niveles procesados.
        d += 1

        # Reduce la longitud de la secuencia a la mitad mediante max-pooling.
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    # Si la secuencia se ha reducido a longitud 1.
    if z1.size(1) == 1:
        if alpha != 0:
            # Calcula la pérdida de instancia.
            loss+= alpha * instance_contrastive_loss(
                z1=z1,
                z2=z2
            )
        # Se cuenta este último nivel.
        d+= 1

    # Normalización final por el número de escalas aplicadas.
    return loss / d