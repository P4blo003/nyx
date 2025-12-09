# ==========================================================================================
# Author: Pablo González García.
# Created: 07/12/2025
# Last edited: 07/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import time
from typing import Dict, List
# Externos:
import torch
from transformers import ZeroShotClassificationPipeline
from transformers import pipeline


# ==============================
# CONSTANTES
# ==============================

ACTIVATION_THRESHOLD:float = 0.70
CANDIDATE_MAP:Dict[str, str] = {
    'INTENT_SQL':'Consultar datos operativos como consumos, temperatura, historial, etc.',
    'INTENT_RAG':'Obtener información de manuales, especificaciones, procedimientos, etc.',
    'INTENT_ANOMALY': 'Predecir fallos, detectar tendencias inusuales o anomalías.',
    'INTENT_CHAT': 'Preguntas generales no técnicas.'
}


# ==============================
# CLASES
# ==============================

class IntentClassifier:
    """
    Clasificador de intención. Se encarga de clasificar la/s intencion/es a partir de una
    pregunta dada.
    """
    # ---- Default ---- #

    def __init__(
        self,
        uri:str,
        candidate_map:Dict[str,str]|None = None,
        activation_threshold:float = 0.7
    ) -> None:
        """
        Inicializa el clasificador.

        Args:
            uri (str): Identificador del modelo a emplear. Puede ser tanto un identificador
                para descargar dede `Hugging Face` como para usar un modelo instalado localmente.
            candidate_map (Dict[str,str]|None): Diccionario con los candidatos. Contiene los posibles
                tipos de la clasificación y una breve descripción.
        """
        # Inicializa las propiedades.
        self.uri:str = uri
        self.candidate_map:Dict[str,str] = candidate_map if candidate_map is not None else CANDIDATE_MAP
        self.labels:List[str] = list(self.candidate_map.keys())
        self.threshold:float = activation_threshold
        
        # Inicializa el modelo.
        self.model:ZeroShotClassificationPipeline = pipeline(
            task='zero-shot-classification',
            model=uri,
            device = 0 if torch.cuda.is_available() else -1
        )


    # ---- Métodos ---- #

    def classify(
        self,
        query:str
    ) -> Dict[str, float]:
        """
        Clasifica la consulta `query` y devuelve una lista de las intenciones
        activadas.

        Args:
            query: Pregunta a clasificar.
        
        Returns:
            Dict[str, float]: Diccionario con las intenciones y sus respectivas puntuaciones.
        """
        # Obtiene el tiempo inicial.
        start:float = time.time()

        # Inferencia zero-shot. Pasa la consulta y las descripciones.
        results:Dict = self.model(                                      # type:ignore
            query,
            list(self.candidate_map.values()),
            multi_label=True
        )

        # Obtiene la duración.
        classification_duration:float = time.time() - start

        # Lista para almacenar las intenciones.
        active_intents:Dict[str, float] = {}

        # Añade la duración de clasificación.
        active_intents['duration'] = classification_duration

        for label_description, score in zip(results['labels'], results['scores']):
            # Busca la clave a partir de la descripción.
            intent:str|None = next((k for k, v in self.candidate_map.items() if v == label_description), None)

            # Comprueba si la puntuación es mayor que el umbral.
            if intent is not None and score >= self.threshold:
                # Añade la intención.
                active_intents[intent] = score

        return active_intents