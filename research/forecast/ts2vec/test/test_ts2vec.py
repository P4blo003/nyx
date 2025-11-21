# ==========================================================================================
# Author: Pablo González García.
# Created: 21/11/2025
# Last edited: 21/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Externos:
import pytest
# Internos:
from src.main import TS2Vec


# ==============================
# TESTS
# ==============================

def test_create_ts2vec_model() -> None:
    """
    Crea un modelo TS2Vec simple.
    """
    # ---- Argumentos ---- #

    # Inicializa los parámetros.
    input_dims:int = 3
    output_dims:int = 3
    
    # ---- Test ---- #

    # Try-Except para manejo de errores.
    try:
        # Crea el modelo.
        model:TS2Vec = TS2Vec(
            input_dims=input_dims,
            output_dims=output_dims
        )
    
    # Si ocurre algún error.
    except Exception as ex:
        # Muestra el error.
        pytest.fail(f"Test fallido. No se pudo crear el modelo => {ex}")

def test_save_model() -> None:
    """
    Almacena el modelo en un documento.
    """
    # ---- Argumentos ---- #

    # Inicializa los parámetros.
    input_dims:int = 3
    output_dims:int = 3
    file_name:str = "test_model.pt"
    
    # ---- Test ---- #

    # Try-Except para manejo de errores.
    try:
        # Crea el modelo.
        model:TS2Vec = TS2Vec(
            input_dims=input_dims,
            output_dims=output_dims
        )

        # Almacena el modelo.
        model.save(file_name=file_name)
    
    # Si ocurre algún error.
    except Exception as ex:
        # Muestra el error.
        pytest.fail(f"Test fallido. No se pudo almacenar el modelo. => {ex}")

def test_load_model() -> None:
    """
    Almacena el modelo en un documento.
    """
    # ---- Argumentos ---- #

    # Inicializa los parámetros.
    input_dims:int = 3
    output_dims:int = 3
    file_name:str = "test_model.pt"
    
    # ---- Test ---- #

    # Try-Except para manejo de errores.
    try:
        # Crea el modelo.
        model:TS2Vec = TS2Vec(
            input_dims=input_dims,
            output_dims=output_dims
        )

        # Almacena el modelo.
        model.load(file_name=file_name)
    
    # Si ocurre algún error.
    except Exception as ex:
        # Muestra el error.
        pytest.fail(f"Test fallido. No se pudo cargar el modelo. => {ex}")