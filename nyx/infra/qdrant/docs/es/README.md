<div align="center">

# Qdrant - Base de Datos Vectorial

![Docker](https://img.shields.io/badge/Docker-24.0-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-1.16.2-green)

</div>


## Tabla de Contenidos

1. [Descripción](#descripción)
2. [Requisitos previos](#requisitos-previos)
3. [Preparación del entorno](#preparación-del-entorno)
4. [Ejecución del servicio](#ejecución-del-servicio)
5. [Conexión a Qdrant](#conexión-a-qdrant)

---

## Descripción

Este proyecto proporciona una infraestructura completa para desplegar **Qdrant**, una base de datos vectorial de alto rendimiento. Qdrant está diseñado para búsquedas de similitud vectorial y es ideal para aplicaciones de Machine Learning, búsqueda semántica y sistemas de recomendación.


### Características principales

- ✅ Despliegue automatizado con Docker Compose.
- ✅ Configuración personalizable mediante variables de entorno.
- ✅ Persistencia de datos mediante volúmenes Docker.
- ✅ Autenticación mediante API Key.
- ✅ Healthcheck integrado.
- ✅ Límites de recursos (CPU/Memoria).
- ✅ Soporte para HTTP y gRPC.

---

## Requisitos previos

Antes de comenzar, asegúrese de tener instalado:

- **Docker** (versión 20.10 o superior)
- **Docker Compose** (versión 2.0 o superior)
- **Make** (opcional, pero recomendado para facilitar comandos)

### Verificar instalación

```bash
# Verificar Docker
docker --version

# Verificar Docker Compose
docker compose version

# Verificar Make (opcional)
make --version
```

---

## Preparación del entorno

### 1. Configurar variables de entorno

El proyecto utiliza variables de entorno para configurar Qdrant.

1. **Copiar el archivo de ejemplo:**

```bash
cp .env.example .env
```

2. **Modificar el archivo `.env`:**

Abra el archivo `.env` y configure las siguientes variables:

```bash
# Versión de Qdrant
QDRANT_VERSION=v1.16.2

# Puerto HTTP (por defecto 6333)
QDRANT_HTTP_PORT=6333

# Puerto gRPC (por defecto 6334)
QDRANT_GRPC_PORT=6334

# API Key para autenticación (IMPORTANTE: cambiar en producción)
QDRANT_API_KEY=tu-api-key-segura-aqui
```

> [!WARNING]
> **Seguridad**: Asegúrese de cambiar `QDRANT_API_KEY` a un valor seguro antes de desplegar en producción. No uses `change-me` en entornos reales.

### 2. Estructura de directorios

El proyecto creará automáticamente los siguientes directorios:

- `data/`: Almacena los datos persistentes de Qdrant.
- `config/`: Contiene la configuración de Qdrant.

Estos directorios se crean automáticamente al ejecutar el servicio.

---

## Ejecución del servicio

### Método 1: Usando Make (Recomendado)

Si tiene Make instalado, puedes usar los siguientes comandos:

```bash
# Iniciar el servicio
make up

# Ver logs en tiempo real
make logs

# Ver estado del servicio
make status

# Detener el servicio (mantiene los datos)
make down

# Limpiar todo (¡CUIDADO! Elimina los datos)
make clean
```

### Método 2: Usando Docker Compose directamente

Si no tiene Make, puede usar Docker Compose directamente:

```bash
# Iniciar el servicio
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f

# Ver estado del servicio
docker compose ps

# Detener el servicio
docker compose down

# Limpiar todo (incluyendo volúmenes)
docker compose down -v
```

### Verificar que el servicio esté corriendo

Después de iniciar el servicio, verifique que esté funcionando correctamente:

```bash
# Opción 1: Ver estado
make status

# Opción 2: Comprobar logs
make logs

# Opción 3: Verificar el healthcheck
docker inspect qdrant --format='{{json .State.Health}}'
```

El servicio está listo cuando vea en los logs:

```
Qdrant HTTP API is listening on 0.0.0.0:6333
Qdrant gRPC API is listening on 0.0.0.0:6334
```

---

## Conexión a Qdrant

Una vez que el servicio esté ejecutándose, puede conectarse a Qdrant de varias maneras:

### 1. Interfaz web (Dashboard)

Qdrant incluye una interfaz web integrada:

```
http://localhost:6333/dashboard
```

> [!NOTE]
> Necesitarás incluir el API Key en las peticiones para autenticarte.