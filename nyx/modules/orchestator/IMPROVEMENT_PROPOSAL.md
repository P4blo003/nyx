# Propuesta de Mejora: Módulo Orchestrator

## 1. Evaluación General

El estado actual del módulo presenta una base estructural limpia utilizando FastAPI y WebSockets. Sin embargo, para un entorno "industrial-critical" y escalable, existen áreas significativas de mejora en cuanto a desacoplamiento, rendimiento y robustez.

## 2. Implementación y Diseño (SOLID & Arquitectura)

### Analisis

Actualmente, la clase `ClientSession` (en `session/client.py`) actúa como un "God Object". Viola el **Principio de Responsabilidad Única (SRP)** porque maneja:

1.  El ciclo de vida de la conexión.
2.  La instanciación de componentes de transporte (receiver/sender).
3.  La gestión del Bus de Eventos.
4.  La coordinación de controladores.

Además, existe un acoplamiento fuerte (**Violación del DIP**) con implementaciones concretas como `EventBus`, `ReceiveLoop` y `SenderLoop` dentro de `initialize`.

### Mejoras Propuestas

1.  **Inyección de Dependencias (DI)**:

    - **Cambio**: No instanciar `EventBus`, `ReceiveLoop` o `SenderLoop` dentro de `ClientSession`. Inyectarlos o usar un contenedor de DI (como `dependency-injector` o el propio de FastAPI si se estructura bien).
    - **Por qué**: Facilita el testing unitario (mocking) y permite cambiar implementaciones de transporte (ej. cambiar WebSocket por gRPC o colas) sin tocar la lógica de sesión.

2.  **Separación de Responsabilidades**:

    - **Cambio**: Crear un `SessionManager` global que gestione la vida de las `ClientSession`. La `ClientSession` debe ser un mero contenedor de estado y contexto, no el orquestador de componentes de infraestructura.
    - **Cambio**: Mover la lógica de "enrutamiento" de mensajes a un `MessageDispatcher` dedicado, en lugar de que el `ReceiveLoop` publique ciegamente o la sesión coordine.

3.  **Abstracción de Eventos**:
    - **Cambio**: Definir eventos tipados (Dataclasses/Pydantic models) en lugar de cadenas de texto mágicas (`"app.close"`, `"ws.send"`).
    - **Por qué**: Evita errores de tipografía (typos) y facilita el autocompletado y refactorización.

## 3. Eficiencia y Rendimiento

### Mejoras Propuestas

1.  **Serialización JSON de Alto Rendimiento**:

    - **Cambio**: Reemplazar la librería estándar `json` por `orjson`.
    - **Impacto**: `orjson` es capaz de serializar/deserializar JSON x10-x20 veces más rápido que la librería estándar. En un sistema basado en mensajes de texto (LLM), esto es crítico.
    - **Implementación**: Usar `orjson` en el `FastApiWebSocketAdapter` y en cualquier transporte de datos.

2.  **Gestión de Tareas (Fire-and-Forget)**:

    - **Cambio**: Al enviar mensajes al LLM o servicios externos, no bloquear el loop de recepción. Usar `asyncio.create_task` con un mecanismo de seguimiento (como `WeakSet` de tareas) para evitar que el GC las elimine prematuramente, pero sin detener el procesamiento de nuevos mensajes.

3.  **Uvicorn Loop con `uvloop`**:
    - **Cambio**: Asegurar que `uvicorn` usa `uvloop` (standard en Linux/Mac, pero verificar instalación).
    - **Impacto**: Mejora significativa en el throughput de E/S.

## 4. Gestión de Errores y Robustez

### Analisis

Actualmente se usa `print(f"Fatal error: {ex}")` y bloques `try-except` genéricos. Esto es insuficiente para producción.

### Mejoras Propuestas

1.  **Logger Estructurado**:

    - **Cambio**: Usar `structlog` o configurar el `logging` estándar para emitir logs en formato JSON.
    - **Por qué**: Permite ingestión directa en sistemas como ELK, Datadog o CloudWatch. Los `print` se pierden o no tienen contexto (timestamp, nivel, trace_id).

2.  **Graceful Shutdown Robusto**:

    - **Cambio**: El manejo de señales actual es manual. FastAPI ya maneja `SIGINT`/`SIGTERM`.
    - **Mejora**: Implementar un "Drain Mode". Al recibir la señal de parada:
      1.  La puerta de enlace (Load Balancer) deja de enviar tráfico.
      2.  El servidor cierra conexiones inactivas.
      3.  El servidor espera (con timeout) a que las sesiones activas terminen su "frase" o procesamiento crítico antes de cortar.

3.  **Circuit Breaker (Patrón)**:
    - **Cambio**: Implementar Circuit Breaker para las llamadas a _SQL Service_, _Rag Service_ y _LLM Service_.
    - **Por qué**: Si el servicio SQL cae, el Orchestrator no debe seguir enviándole peticiones y saturando hilos de espera. Debe "abrir el circuito" y fallar rápido retornando error al cliente inmediatamente.

## 5. Problemas Futuros y Escalabilidad

1.  **Estado en Memoria (Stateful)**:

    - **Problema**: El `ClientSession` guarda el estado en la memoria del proceso (RAM).
    - **Riesgo**: Si despliegas 2 réplicas del Orchestrator tras un Nginx Load Balancer, y el WebSocket se reconecta, podría caer en el otro servidor y perder el contexto.
    - **Solución Futura**: Externalizar el estado de la sesión (Session Store) a **Redis**.

2.  **Cuellos de Botella en el Bus de Eventos**:

    - **Problema**: El `EventBus` actual parece ser en memoria (in-process).
    - **Riesgo**: Si un proceso pesado bloquea el bus, todo se detiene.
    - **Solución Futura**: Si se escala a microservicios reales (procesos separados), mover el bus a **RabbitMQ** o **NATS**.

3.  **Seguridad**:
    - **Ausencia**: No se ve validación de tokens en el establecimiento del WebSocket.
    - **Mejora**: Middleware de autenticación que valide JWT antes de aceptar el `websocket.accept()`.
