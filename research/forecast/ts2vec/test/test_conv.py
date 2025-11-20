# ==========================================================================================
# Author: Pablo González García.
# Created: 19/11/2025
# Last edited: 19/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import warnings
import logging
from typing import List
# Externos:
import torch
from torch import nn
import pytest
# Internos:
from src.conv import SameConv1d, SameConv1dBlock


# ==============================
# TESTS
# ==============================

def test_same_conv1_shapes_and_grad(
    device:str='cuda',
    batch_size:int = 4,
    gradient_test_tolerance:float = 1e-6
) -> None:
    """
    Test unitario para la clase SameConv1d.

    Verifica que:
    1. La salida mantiene la misma longitud temporal que la entrada.
    2. Los canales de salida son correctos.
    3. La capa es compatible con forward + backward (gradientes).
    4. Funciona con distintos kernels y dilataciones.

    Args:
        device (str): Dispositivo donde se ejecuta.
    """
    # Crea el logger para mostrar información.
    logger:logging.Logger = logging.getLogger('test_same_conv1_shapes_and_grad')

    # Configuración del test.
    test_config:List = [
        # Configuraciones básicas.
        (1, 1, 10, 3, 1),
        (3, 5, 20, 4, 1),
        (2, 8, 50, 5, 2),
        # Otros casos.
        (1, 10, 1, 3, 1),       # Mínima longitud de sequencia.
        (8, 8, 100, 1, 1),      # Tamaño de kernel 1.
        (4, 4, 15, 7, 3)        # kernel grande con dilatación.
    ]

    # Itera sobre las diferentes configuraciones.
    for i, (in_channels, out_channels, sequence_length, kernel_size, dilation) in enumerate(test_config):
        # Genera el nombre del test.
        test_name:str = f"config_{i}-in_{in_channels}-out_{out_channels}-len_{sequence_length}-k_{kernel_size}-d_{dilation}"

        # Try-Except para manejo de errores.
        try:
            # Inicializa la capa.
            conv:SameConv1d = SameConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation
            ).to(device=device)

            # Genera el tensor.
            x:torch.Tensor = torch.randn(
                batch_size,
                in_channels,
                sequence_length,
                device=device,
                requires_grad=True,
                dtype=torch.float32
            )

            # Realiza la convolución.
            with warnings.catch_warnings():
                # Trata los warnings como errores.
                warnings.simplefilter("error")
                # Realiza la convolución.
                y:torch.Tensor = conv.forward(x=x)
            
            # ---- Forma ---- #

            # Realiza la prueba.
            assert y.shape == (batch_size, out_channels, sequence_length), (
                f"Fallo shape en test: {test_name}."
                f"Esperado: {(batch_size, out_channels, sequence_length)}, Obtenido: {y.shape}"
            )


            # ---- Gradiente ---- #

            loss:torch.Tensor = y.sum()
            loss.backward()

            # Realiza las pruebas.
            assert x.grad is not None, f"Gradientes no establecidos en test: {test_name}"
            assert torch.isfinite(x.grad).all(), f"Gradientes no finitos en el input para test: {test_name}"

            # Reecorre todos los parámetros entrenables de la capa (pesos y bias)
            for param_name, param in conv.named_parameters():
                # Realiza las pruebas.
                assert param.grad is not None, f"Gradientes para {param_name} no calculados para test: {test_name}"
                assert torch.isfinite(param.grad).all(), f"Gradientes no finitos en {param_name} para test: {test_name}"

                # Calccula la norma L2 del gradiente del parámetro.
                grad_norm = param.grad.norm()
                # Realiza la prueba.
                assert grad_norm > gradient_test_tolerance, (
                    f"Gradientes desvanecidos en {param_name} para test: {test_name}"
                    f"Norma del gradiente: {grad_norm:.2e}"
                )

            # Resetea los gradientes para la próxima iteración.
            conv.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            

            # ---- Final ---- #

            # Imprime test pasado.
            logger.info(f"✓ {test_name}\tpassed")

        # Si ocurre algún error.
        except Exception as e:
            # Imprime el error.
            pytest.fail(f"Test fallido para test: {test_name} => {str(e)}")

def test_same_conv1_block_shapes_and_grad(
    device:str='cuda',
    batch_size:int = 4,
    gradient_test_tolerance:float = 1e-6   
) -> None:
    """
    Test unitario para la clase SameConv1dBlock.

    Verifica que:
    1. La salida mantiene la misma longitud temporal que la entrada.
    2. Los canales de salida son correctos.
    3. El bloque es compatible con forward + backward (gradientes).
    4. Funciona con distintos kernels, dilataciones y configuraciones finales.
    5. La conexión residual funciona correctamente.

    Args:
        device (str): Dispositivo donde se ejecuta.
        batch_size (int): Tamaño del batch para testing.
        gradient_test_tolerance (float): Tolerancia para detección de gradientes.
    """
    # Crea el logger para mostrar información.
    logger: logging.Logger = logging.getLogger('test_same_conv1d_block_shapes_and_grad')

    # Configuración del test.
    test_config: List = [
        # Configuraciones básicas - sin proyector (mismos channels, no final)
        (4, 4, 10, 3, 1, False),
        (8, 8, 20, 5, 1, False),
        (4, 4, 50, 3, 2, False),
        
        # Configuraciones con proyector (diferentes channels)
        (4, 8, 15, 3, 1, False),
        (8, 4, 25, 5, 1, False),
        (3, 16, 30, 7, 1, False),
        
        # Configuraciones finales (siempre con proyector)
        (4, 4, 10, 3, 1, True),
        (4, 8, 20, 5, 1, True),
        (8, 4, 15, 3, 2, True),
        
        # Casos límite
        (1, 1, 5, 3, 1, False),      # Mínimo de canales
        (16, 16, 100, 1, 1, False),  # Kernel size 1
        (4, 4, 10, 11, 3, False),    # Kernel grande con dilatación
    ]

    # Itera sobre las diferentes configuraciones.
    for i, (in_channels, out_channels, sequence_length, kernel_size, dilation, final) in enumerate(test_config):
        # Genera el nombre del test.
        test_name: str = f"config_{i}-in_{in_channels}-out_{out_channels}-len_{sequence_length}-k_{kernel_size}-d_{dilation}-f_{final}"

         # Try-Except para manejo de errores.
        try:
            # Inicializa el bloque.
            block: SameConv1dBlock = SameConv1dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                final=final
            ).to(device=device)

            # Genera el tensor de entrada.
            x: torch.Tensor = torch.randn(
                batch_size,
                in_channels,
                sequence_length,
                device=device,
                requires_grad=True,
                dtype=torch.float32
            )

            # Realiza el forward pass.
            with warnings.catch_warnings():
                # Trata los warnings como errores.
                warnings.simplefilter("error")
                # Realiza la forward pass del bloque.
                y: torch.Tensor = block.forward(x=x)
            
            
            # ---- Forma ---- #

            # Realiza la prueba.
            assert y.shape == (batch_size,  out_channels, sequence_length), (
                f"Fallo shape en test: {test_name}"
                f"Esperado: {(batch_size, out_channels, sequence_length)}, Obtenido: {y.shape}"
            )


            # ---- Conexión Residual ---- #

            # Realiza las pruebas.
            assert hasattr(block, 'conv1'), f"Bloque no tiene conv1 en test: {test_name}"
            assert hasattr(block, 'conv2'), f"Bloque no tiene conv2 en test: {test_name}"

            # Verifica el projector según la configuración.
            if in_channels != out_channels or final:
                # Realiza las pruebas.
                assert block.projector is not None, (
                    f"Projector debería existir para test: {test_name} "
                    f"(in_channels != out_channels or final=True)"
                )
                assert isinstance(block.projector, nn.Conv1d), (
                    f"Projector no es Conv1d en test: {test_name}"
                )
            else:
                assert block.projector is None, (
                    f"Projector no debería existir para test: {test_name} "
                    f"(in_channels == out_channels and final=False)"
                )
            

            # ---- Gradiente ---- #

            # TODO


            # ---- Final ---- #

            # Imprime test pasado.
            logger.info(f"✓ {test_name}\tpassed")
        
        # Si ocurre algún error.
        except Exception as e:
            # Imprime el error.
            pytest.fail(f"Test fallido para test: {test_name} => {str(e)}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Establece la configuración.
    logging.basicConfig(level=logging.INFO)

    # Ejecuta el test para una capa convolucionaal.
    test_same_conv1_shapes_and_grad(device='cuda')
    # Ejecuta el test para una capa convolucional.
    test_same_conv1_block_shapes_and_grad(device='cuda')