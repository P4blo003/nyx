# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import json
from datetime import datetime
from typing import List
from pathlib import Path
# Externos:
from pydantic import BaseModel, Field


# ==============================
# CLASES
# ==============================

class TrainingData(BaseModel):
    """
    Almacena la información del entrenamiento.

    Attributes:
        train_size (int): Número de entradas para el entrenamiento.
        valid_size (int): Número de entradas para la validación.
        test_size (int): Número de entradas para la prueba.
        output_dims (int): Tamaño de los embeddings generados.
        hidden_dims (int): Dimensión oculta del encoder.
        depth (int): Número de bloques residuales del encoder.
        learning_rate (float): Ratio de aprendizaje.
        batch_size (int): Tamaño del batch.
        max_train_length (int): Máxima longitud de la secuencia para entrenamiento.
        n_iters (int): Número de iteraciones del entrenamiento.
        n_epochs (int): Número de épocas del entrenamiento.
        duration (float): Duración en segundos del entrenamiento.
        losses (List[float]): Listado con las losses del entrenamiento.
    """
    # ---- Atributos ---- #
    # Información del conjunto de datos.
    train_size:int = Field(ge=0)
    valid_size:int = Field(ge=0)
    test_size:int = Field(ge=0)
    # Información del modelo.
    output_dims:int = Field(ge=0)
    hidden_dims:int = Field(ge=0)
    depth:int = Field(ge=0)
    learning_rate:float = Field(ge=0.0)
    batch_size:int = Field(ge=0)
    max_train_length:int = Field(ge=0)
    # Información del entrenamiento.
    n_iters:int = Field(ge=0)
    n_epochs:int = Field(ge=0)
    duration:float = Field(ge=0.0)
    losses:List[float]


    # ---- Métodos ---- #

    def to_json(
        self,
        file_path:str,
        include_metadata:bool = True
    ) -> str:
        """
        Convierte la instancia a Json.

        Args:
            file_path (str): Ruta donde almacenar el archivo.
            include_metadata (bool): Incluir metadátos.
        Returns:
            str: Representación JSON del objeto.
        """
        # Crea diccionario con los datos
        data = self.model_dump()
        
        # Agregar metadatos si se solicita
        if include_metadata:
            metadata = {
                "_metadata": {
                    "version": "1.0",
                    "exported_at": datetime.now().isoformat(),
                    "class_name": self.__class__.__name__,
                    "fields_count": len(data)
                }
            }
            # Actualiza los datos.
            data.update(metadata)
        
        # Convierte a Json.
        json_str:str = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )

        # Guarda el fichero.
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            # Escribe el fichero.
            file.write(json_str)
        
        # Retorna el formato json.
        return json_str