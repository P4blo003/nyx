# ==========================================================================================
# Author: Pablo González García.
# Created: 07/12/2025
# Last edited: 07/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import Dict
# Externas:
from fastapi import APIRouter, Depends, HTTPException, status
# Internas:
from model.classifier import IntentClassifier


# ==============================
# CONSTANTES
# ==============================

# Inicializa la ruta para este módulo.
ROUTER:APIRouter = APIRouter()


# ==============================
# RUTAS
# ==============================

@ROUTER.post("/classify")
async def classify_query(
    query:str,
    classifier:IntentClassifier = Depends()
) -> Dict[str,float]:
    """
    Clasifica la consulta `query` y retorna un diccionario con las intenciones
    de esta.

    Args:
        query (str): Consulta a clasificar.
        classifier (IntentClassifier): Clasificador de consultas.
    
    Returns:
        Dict[str,float]: Diccionario con las intenciones y sus puntuaciones.
    """
    # Try-Except para manejo de errores.
    try:
        # Retorna las clasificaciones.
        return classifier.classify(query=query)
    
    # Si ocurre algún error durante la ejecución
    except RuntimeError as ex:
        # Lanza error 500.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia del clasificador: {str(ex)}"
        )

    # Si ocurre un error genérico.
    except Exception as ex:
        # Cualquier otro error no manejado
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error desconocido durante la clasificación: {str(ex)}"
        )