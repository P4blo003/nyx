# ==========================================================================================
# Author: Pablo González García.
# Created: 07/12/2025
# Last edited: 07/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
from typing import List
# Internos:
from model.classifier import IntentClassifier


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    
    # Inicializa el clasificador.
    classifier:IntentClassifier = IntentClassifier(
        uri='facebook/bart-large-mnli'
    )

    # Querys:
    querys:List[str] = [
        "¿Cuál es el valor actual de presión en el reactor 4?",     # INTENT_SQL.
        "¿Qué dice el manual sobre reparación de la turbina T-3?",  # INTENT_RAG.
        "¿Hay algo raro en la temperatura de la caldera? Predice la desviación.",   # INTENT_ANOMALY
        "Dame el historial de fallos y el procedimiento de emergencia.",
        "Hola, ¿Cómo estas hoy?."
    ]

    # Clasifica las querys.
    for i, query in enumerate(querys):
        # Imprime información.
        print(f"=== Consulta [{i}] ===")
        # Clasifica la consulta.
        print(classifier.classify(query=query))

