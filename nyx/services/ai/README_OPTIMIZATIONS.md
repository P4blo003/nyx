# Análisis y Propuesta de Mejoras - AI Service

Este documento detalla un análisis exhaustivo del servicio de IA actual, identificando puntos críticos y proponiendo correcciones para elevar el código a un estándar industrial, eficiente y escalable.

## 📊 Estado Actual

El servicio actualmente funciona como un prototipo funcional pero presenta carencias críticas para un entorno de producción de alto rendimiento.

*   **Puntos Fuertes:** Estructura de carpetas limpia (DDD-like), uso de `FastAPI` y `Triton Inference Server` vía gRPC.
*   **Puntos Críticos:** Ineficiencia severa en la resolución de modelos, acoplamiento fuerte a un caso de uso específico ("EMBEDDING"), falta de observabilidad y robustez.

## 🚀 Optimizaciones y Correcciones Críticas

### 1. Eficiencia y Rendimiento (Prioridad Alta)

El problema más grave detectado es el **descubrimiento de modelos en cada petición**.

*   **Problema:** En `InferenceService.make_infer`, el código itera sobre todos los clientes y solicita la lista de modelos (`get_models`) al servidor Triton en **cada inferencia**.
*   **Impacto:** Latencia innecesaria (N llamadas de red extra por petición), sobrecarga en el servidor Triton.
*   **Solución:**
    *   Implementar un **Model Discovery al inicio** (startup).
    *   Mantener un **mapa en memoria** (Cache) de `nombre_modelo -> cliente_triton`.
    *   Actualizar este mapa periódicamente (background task) o bajo demanda, no en el "hot path" de inferencia.

### 2. Arquitectura y Principios SOLID

#### S - Single Responsibility Principle (SRP)
*   **Problema:** `TritonSDK` mezcla la lógica de cliente genérico con la lógica específica de embeddings (`inputs=[InferInput(name="TEXT"...)]`).
*   **Corrección:**
    *   `TritonSDK` debe ser agnóstico al modelo (recibir inputs/outputs genéricos).
    *   Crear **Estrategias/Adaptadores** específicos por tipo de modelo (ej: `EmbeddingModelAdapter`, `GenerativeModelAdapter`) que sepan cómo formatear los tensores para modelos concretos (ej. BERT, Llama, etc.).

#### O - Open/Closed Principle (OCP)
*   **Problema:** Si quieres añadir un modelo de clasificación de imágenes, tienes que modificar `TritonSDK.make_infer`.
*   **Corrección:** Al usar adaptadores, puedes añadir nuevos tipos de modelos sin tocar el código base del SDK.

#### L - Liskov Substitution Principle (LSP)
*   **Problema:** `InferenceService.make_infer` acepta un parámetro `texts`, asumiendo que siempre es texto. Si el servicio de IA evoluciona a imágenes, la interfaz se rompe.
*   **Corrección:** Definir DTOs de entrada genéricos o específicos por Tarea (ej. `TextInferenceRequest`, `ImageInferenceRequest`) y usar Generics o Union Types correctamente.

### 3. Corrección de Lógica de Negocio (Bug Crítico)

*   **Ubicación:** `src/application/services/inference_service.py`
*   **Problema:** La variable `model_name` está **hardcodeada** a `"bge_m3_ensemble"`.
*   **Consecuencia:** El parámetro `task` de la URL (`/inference/{task}`) es ignorado. El sistema no puede servir múltiples modelos.
*   **Corrección:** Usar el parámetro `task` para buscar en el mapa de modelos (mencionado en el punto 1) el modelo correcto a invocar.

### 4. Robustez Industrial

*   **Manejo de Errores:**
    *   Implementar un **Global Exception Handler** en FastAPI para capturar errores de gRPC y devolver códigos HTTP semánticos (ej. 503 si Triton está caído, 404 si el modelo no existe).
    *   Envolver llamadas gRPC con **Retries** (reintentos exponenciales) para fallos transitorios de red.
*   **Circuit Breaker:** Si un servidor Triton falla repetidamente, dejar de enviarle peticiones temporalmente para evitar cascadas de fallos.
*   **Observabilidad:**
    *   Añadir métricas (Prometheus) para: Latencia de inferencia, Tasa de errores, Uso de GPU (vía métricas de Triton re-expuestas).
    *   **Structured Logging:** Usar logs JSON para facilitar la ingestión en sistemas como ELK/Datadog, incluyendo `request_id` para trazabilidad.

### 5. Configuración y Seguridad

*   **Configuración:** Usar `Pydantic Settings` para validar variables de entorno al inicio. Actualmente se hace con `os.environ.get` disperso o básico.
*   **Metadatos innecesarios:** Eliminar cabeceras de autor (Author, Created...) de cada archivo. Git ya gestiona esa historia. "Clean Code".

## 📋 Plan de Implementación Recomendado

1.  **Refactorizar `TritonSDK`**: Hacerlo genérico.
2.  **Crear `ModelRegistry`**: Singleton que carga y cachea `task -> model_config` al inicio.
3.  **Corregir `InferenceService`**: Usar el registry y respetar el parámetro `task`.
4.  **Middleware de Logs y Errores**: Estandarizar respuestas.
5.  **Dockeritzación**: Asegurar un Dockerfile Multi-stage para reducir el tamaño de la imagen final.

---
**Nota:** Estas correcciones transformarían el servicio de un "script con FastAPI" a un **Microservicio Robusto**.
