<div align="center">

# TS2Vec

---

![Python](https://img.shields.io/badge/Python-3.13.5-blue)

---

</div>

## Instalación y configuración inicial.

TS2Vec requiere un entorno correctamente aislado y configurado para asegurar compatibilidad con sus dependencias, estabilidad durante las fases de entrenamiento y soporte optimizado para GPU. A continuación se detalla el procedimiento recomendado para reparar el entorno virtual y establecer la base técnica necesaria para ejecutaar el modelo.

### 1. Preparación del entorno virtual.

La creación de un entorno virtual es esencial para evitar conflicto entre dependencias del sistema y las específicas del proyecto. Se recomienda emplear **venv**(incluido en Python>= 3.3) o **virtualenv** para aislar las librerías.

```bash
# Crea el entorno virtual.
python -m venv venv
```

Este comando genera el entorno aislado denominado **venv** que actuará como espacio de ejecución independiente. 

```bash
# Activa el entorno virtual en entornos linux/maxOS.
source venv/bin/activate
```

```bash
# Activa el entorno virtual en entornos Windows.
.\venv\Scripts\activate
```

Una vez activado, el intérprete, las dependencias y las herramientas asociadas se resolverán exclusivamente dentro del entorno, evitando interferencias con el sistema principal.

### 2. Actualizar pip y herramientas base.

Antes de instalar dependencias es recomendable actualizar los gestores del entorno.

```bash
# Actualiza pip.
python -m pip install --upgrade pip setuptools wheel
```

Esta actualización garantiza compatibilidad con paquetes compilados, distribución de binarios y optimizaciones de instalación para librerías con componentes nativos como [PyTorch](https://pytorch.org/).

### 3. Verifica la versión dee Python del entorno.

Para asegurarse de que el entorno esté utilizando la versión correcta (Python 3.13.5).

```bash
# Comprueba la versión de python.
python --version
```

El resultado debe coincidir estrictamente con la versión requerida por el proyecto.

### 4. Instalación de dependencias.

TS2Vec depende de PyTorch y otras librerías de Python para procesamiento de datos, entrenamiento y visualización. Se recomienda instalar todo de forma estructurada.

#### 4.1 PyTorch.

TS2Vec se basa en PyTorch para operaciones tensoriales y soporte GPU. Es importante instalar la versión de PyTorch compatible con tu versión CUDA.

```bash
# Comprueba la versión de CUDA.
nvidia-smi
```

En la parte superior saldrá la información de CUDA. Una vez se sepa la versión, puede dirigirse al siguiente [enlace](https://pytorch.org/get-started/locally/) para configurar el comando completo para instalar PyTorch correctamente. Una vez haya establecido los parámetros, solo debe instalar la librería dentro de su **entorno virtual**.

```bash
# Instala PyTorch. Ejemplo para CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

En caso de no disponer de GPU, instalé la versión CPU-only.
```bash
# Instala PyTorch. Versión CPU-only
pip install torch torchvision torchaudio
```

Puede comprobar de manera rápida la instalación con:

```python
import torch
print("PyTorch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
```