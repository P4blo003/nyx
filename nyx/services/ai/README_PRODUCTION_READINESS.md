# Informe de Preparación para Producción - Nyx AI Service

Este documento detalla los cambios necesarios para elevar el servicio `nyx/services/ai` a un estándar industrial de producción, soportando múltiples clientes, alta escalabilidad (Kubernetes), y observabilidad robusta.

## 1. Evaluación de Estado Actual

### Puntos Fuertes
- **Arquitectura**: El uso de Clean Architecture (Capas: Domain, Application, Infrastructure, Interfaces) es excelente y facilita el mantenimiento y testabilidad.
- **Tipado**: Uso consistente de *Type Hints* de Python.
- **Inyección de Dependencias**: Buen uso de patrones de diseño (Builder, Manager) para desacoplar implementaciones.

### Áreas Críticas de Mejora
- **Configuración**: Valores *hardcoded* (nombres de host, puertos) y dependencia estricta de ficheros locales (`triton_config.yaml`). No sigue "The Twelve-Factor App".
- **Docker**: El `Dockerfile` actual no sigue las mejores prácticas de seguridad (usuario root) ni eficiencia de *cache* de capas.
- **Escalabilidad**: Uso de `InMemoryCache` impide el escalado horizontal consistente (Stateful).
- **Resiliencia**: Falta de gestión avanzada de errores, *retries*, *timeouts* explícitos y *circuit breakers* en las conexiones gRPC con Triton.
- **Observabilidad**: Logs estructurados básicos presentes, pero falta integración con métricas (Prometheus) y trazabilidad distribuida.

---

## 2. Cambios Requeridos (Checklist)

### 2.1. Contenedorización e Infraestructura (`Dockerfile`)
- [ ] **Optimización de Capas**: Modificar `Dockerfile` para copiar `requirements.txt` e instalar dependencias *antes* de copiar el código fuente (`COPY . .`). Esto permite aprovechar la caché de Docker en builds sucesivos.
- [ ] **Seguridad (Non-root user)**: Crear y utilizar un usuario sin privilegios (`appuser`) para ejecutar la aplicación, evitando riesgos de seguridad en el host.
- [ ] **Multi-stage Build**: Utilizar *multi-stage builds* para separar el entorno de construcción del de ejecución, reduciendo el tamaño de la imagen final.
- **Ejemplo**:
  ```dockerfile
  FROM python:3.13-slim as builder
  WORKDIR /build
  COPY requirements.txt .
  RUN pip install --user -r requirements.txt

  FROM python:3.13-slim
  WORKDIR /app
  COPY --from=builder /root/.local /root/.local
  COPY src/ .
  ENV PATH=/root/.local/bin:$PATH
  USER appuser
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

### 2.2. Gestión de Configuración (`config/` & `src/infrastructure`)
- [ ] **Configuración por Variables de Entorno**: Eliminar la dependencia estricta del fichero `triton_config.yaml`.
    - Modificar `Config` (Pydantic) para leer de variables de entorno (e.g., `TRITON_HOST_ONNX`, `TRITON_PORT_GRPC`).
    - Permitir `triton_config.yaml` como *fallback* opcional, no obligatorio.
- [ ] **Eliminar Hardcoding**:
    - En `main.py`, eliminar `client_class='grpc'` hardcoded. Permitir configurarlo por entorno (`TRITON_CLIENT_TYPE=grpc`).
    - Permitir definir dinámicamente la lista de servidores Triton, no solo `triton-onnx` y `triton-vllm`.

### 2.3. Código Fuente y Lógica de Negocio (`src/`)

#### Escalabilidad y Caché
- [ ] **Caché Distribuida (Redis)**: Reemplazar `InMemoryCache` por una implementación de `ICache` basada en **Redis**.
    - **Por qué**: En Kubernetes, si tienes 10 réplicas del servicio AI, cada una tendría su propia caché en memoria, provocando inconsistencias y mayor carga a Triton al cargar modelos repetidamente. Redis centraliza el estado de los modelos cargados/descargados.

#### Resiliencia y Conexión con Triton
- [ ] **Timeouts y Retries**: Añadir lógica de *retry* con *exponential backoff* en `GrpcAsyncClient.infer` y otros métodos críticos para tolerar fallos transitorios de red.
- [ ] **Gestión de Excepciones Específica**: Capturar `InferenceServerException` de la librería `tritonclient` y convertirlas a excepciones de dominio propias para un manejo de errores más limpio en capas superiores.
- [ ] **Circuit Breaker**: Implementar un patrón *Circuit Breaker* para dejar de enviar peticiones a un servidor Triton caído y permitir su recuperación.

#### Logging y Observabilidad ( `shared/utilities/log/`)
- [ ] **JSON Logs por Defecto**: En producción, forzar el uso de `JSONFormatter` para todos los logs (configurable via `LOG_FORMAT=json`).
- [ ] **Métricas Prometheus**: Exponer un endpoint `/metrics` utilizando `prometheus-fastapi-instrumentator`.
    - Métricas clave: Latencia de inferencia, tasa de errores por modelo, uso de memoria, estado de conexión a Triton.
- [ ] **Health Checks**: Implementar rutas `/health/live` y `/health/ready`.
    - `Ready`: Debe verificar conexión real con Triton y (opcionalmente) Redis.

### 2.4. Kubernetes Readiness
- [ ] **Liveness/Readiness Probes**: Configurar los probes en el deployment de K8s apuntando a los endpoints de salud creados.
- [ ] **Horizontal Pod Autoscaling (HPA)**: Configurar HPA basado en métricas de CPU o métricas custom (colas de inferencia).

## 3. Resumen de Implementación Prioritaria

1.  **Refactor Configuración**: Migrar a Pydantic Settings con soporte de Env Vars.
2.  **Docker Update**: Seguridad y Caching.
3.  **Redis Cache**: Implementar `RedisCache` en `src/infrastructure/cache/redis.py`.
4.  **Resiliencia Triton**: Añadir retries y gestión de errores en `src/infrastructure/triton/client/grpc.py`.
