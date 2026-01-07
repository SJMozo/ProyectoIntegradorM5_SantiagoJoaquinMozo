# 📚 Guía de Librerías y Dependencias del Proyecto

Este documento detalla todas las librerías necesarias para ejecutar el proyecto de predicción de pagos a tiempo, con sus versiones recomendadas, propósito y comandos de instalación.

---

## 📋 Tabla de Contenidos

1. [Instalación Rápida](#-instalación-rápida)
2. [Librerías Core de Machine Learning](#-librerías-core-de-machine-learning)
3. [Librerías de Visualización](#-librerías-de-visualización)
4. [Librerías para API y Despliegue](#-librerías-para-api-y-despliegue)
5. [Librerías de Monitoreo](#-librerías-de-monitoreo)
6. [Requisitos del Sistema](#-requisitos-del-sistema)
7. [Troubleshooting](#-troubleshooting)

---

## 🚀 Instalación Rápida

### Opción 1: Instalar todas las dependencias de una vez

```bash
pip install -r requirements.txt
```

### Opción 2: Instalación manual por categorías

Ver secciones detalladas más abajo.

---

## 🤖 Librerías Core de Machine Learning

Estas librerías son fundamentales para el procesamiento de datos y construcción de modelos.

### 1. **NumPy** - Cálculos numéricos
```bash
pip install numpy==1.24.3
```
- **Propósito:** Operaciones con arrays y matrices
- **Usado en:** Todos los módulos
- **Alternativas:** Ninguna (base de todo el ecosistema científico)

### 2. **Pandas** - Manipulación de datos
```bash
pip install pandas==2.0.3
```
- **Propósito:** DataFrames y análisis de datos tabulares
- **Usado en:** Todos los módulos
- **Características clave:**
  - Lectura de Excel (`pd.read_excel`)
  - Manipulación de DataFrames
  - Manejo de datos faltantes

### 3. **Scikit-learn** - Machine Learning tradicional
```bash
pip install scikit-learn==1.3.0
```
- **Propósito:** Algoritmos de ML, preprocesamiento, validación
- **Usado en:** `ft_engineering.py`, `model_training_evaluation.py`
- **Módulos utilizados:**
  - `sklearn.preprocessing`: Escalado, encoding
  - `sklearn.pipeline`: Pipelines de preprocesamiento
  - `sklearn.model_selection`: Train/test split, validación cruzada
  - `sklearn.metrics`: Métricas de evaluación
  - `sklearn.ensemble`: Random Forest
  - `sklearn.linear_model`: Regresión Logística, SGD
  - `sklearn.tree`: Decision Tree
  - `sklearn.svm`: Support Vector Machines
  - `sklearn.naive_bayes`: Naive Bayes
  - `sklearn.neighbors`: KNN
  - `sklearn.discriminant_analysis`: LDA

### 4. **XGBoost** - Gradient Boosting optimizado
```bash
pip install xgboost==2.0.3
```
- **Propósito:** Modelo principal de predicción
- **Usado en:** `model_training_evaluation.py`, `model_deploy.py`
- **¿Por qué XGBoost?**
  - Alto rendimiento en datos tabulares
  - Manejo nativo de valores faltantes
  - Regularización incorporada
  - Rápido entrenamiento

### 5. **OpenPyXL** - Lectura de archivos Excel
```bash
pip install openpyxl==3.1.2
```
- **Propósito:** Backend para `pd.read_excel()`
- **Usado en:** `cargar_datos.py`
- **Nota:** Pandas requiere esta librería para leer .xlsx

---

## 📊 Librerías de Visualización

Para crear gráficas y dashboards interactivos.

### 1. **Matplotlib** - Gráficas estáticas
```bash
pip install matplotlib==3.7.2
```
- **Propósito:** Gráficas básicas (líneas, barras, scatter)
- **Usado en:** `model_training_evaluation.py`
- **Casos de uso:**
  - Curvas de aprendizaje
  - Matrices de confusión
  - ROC curves

### 2. **Seaborn** - Gráficas estadísticas mejoradas
```bash
pip install seaborn==0.12.2
```
- **Propósito:** Visualizaciones estadísticas elegantes
- **Usado en:** `model_training_evaluation.py`
- **Basado en:** Matplotlib (lo extiende)
- **Ventaja:** Estilos profesionales por defecto

### 3. **Plotly** - Gráficas interactivas
```bash
pip install plotly==5.17.0
```
- **Propósito:** Visualizaciones interactivas para web
- **Usado en:** `model_monitoring.py`
- **Características:**
  - Zoom, pan, hover
  - Exportación a HTML
  - Integración con Streamlit

---

## 🌐 Librerías para API y Despliegue

Para servir el modelo como API REST.

### 1. **FastAPI** - Framework web moderno
```bash
pip install fastapi==0.104.1
```
- **Propósito:** Crear API REST de alto rendimiento
- **Usado en:** `model_deploy.py`
- **Ventajas:**
  - Validación automática con Pydantic
  - Documentación interactiva (Swagger UI)
  - Async support
  - Muy rápido (comparable a Node.js)

### 2. **Uvicorn** - Servidor ASGI
```bash
pip install uvicorn==0.24.0
```
- **Propósito:** Servidor para ejecutar FastAPI
- **Usado en:** `model_deploy.py`
- **Comando de ejecución:**
  ```bash
  uvicorn model_deploy:app --reload
  ```

### 3. **Pydantic** - Validación de datos
```bash
pip install pydantic==2.5.0
```
- **Propósito:** Validación y serialización de datos
- **Usado en:** `model_deploy.py`
- **Características:**
  - Type hints enforcement
  - Validación automática
  - JSON schema generation
- **Nota:** Incluido con FastAPI, pero se puede actualizar

### 4. **Requests** - Cliente HTTP
```bash
pip install requests==2.31.0
```
- **Propósito:** Hacer peticiones HTTP a la API
- **Usado en:** `model_monitoring.py`
- **Ejemplo:**
  ```python
  response = requests.post("http://localhost:8000/predict", json=data)
  ```

---

## 📡 Librerías de Monitoreo

Para supervisar el modelo en producción.

### 1. **Streamlit** - Dashboard interactivo
```bash
pip install streamlit==1.29.0
```
- **Propósito:** Crear aplicaciones web de datos sin JavaScript
- **Usado en:** `model_monitoring.py`
- **Características:**
  - Interfaz web automática
  - Widgets interactivos
  - Actualización en tiempo real
- **Comando de ejecución:**
  ```bash
  streamlit run model_monitoring.py
  ```

### 2. **Evidently** - Detección de drift
```bash
pip install evidently==0.4.11
```
- **Propósito:** Monitorear calidad de datos y modelos
- **Usado en:** `model_monitoring.py`
- **Funcionalidades:**
  - Data drift detection
  - Model performance monitoring
  - Reportes visuales
  - Alertas automáticas

---

## 🖥️ Requisitos del Sistema

### Python
- **Versión requerida:** Python 3.8 - 3.11
- **Recomendado:** Python 3.10
- **No soportado:** Python 3.12+ (algunas librerías aún no compatibles)

### Sistema Operativo
- ✅ Windows 10/11
- ✅ macOS 11+
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)

### Hardware mínimo
- **RAM:** 4 GB (8 GB recomendado)
- **Disco:** 2 GB libres
- **CPU:** Cualquier procesador moderno (multi-core preferido)

---

## 📦 Archivo `requirements.txt` Completo

Crea un archivo `requirements.txt` con el siguiente contenido:

```txt
# ==========================================
# CORE DE MACHINE LEARNING
# ==========================================
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
xgboost==2.0.3

# ==========================================
# VISUALIZACIÓN
# ==========================================
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# ==========================================
# API Y DESPLIEGUE
# ==========================================
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
requests==2.31.0

# ==========================================
# MONITOREO
# ==========================================
streamlit==1.29.0
evidently==0.4.11

# ==========================================
# UTILIDADES
# ==========================================
openpyxl==3.1.2        # Para leer archivos Excel
python-multipart==0.0.6  # Para FastAPI file uploads

# ==========================================
# NOTEBOOKS (OPCIONAL)
# ==========================================
jupyter==1.0.0
ipykernel==6.25.2
```

---

## 🔧 Instalación Paso a Paso

### 1. Crear un entorno virtual (RECOMENDADO)

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### En macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Actualizar pip
```bash
pip install --upgrade pip
```

### 3. Instalar todas las dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
pip list
```

---

## 🐛 Troubleshooting

### Problema 1: Error al instalar XGBoost

**Síntomas:**
```
ERROR: Could not build wheels for xgboost
```

**Solución:**
```bash
# Windows: Instalar Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# macOS: Instalar Xcode Command Line Tools
xcode-select --install

# Linux: Instalar build essentials
sudo apt-get install build-essential
```

### Problema 2: Error con OpenPyXL

**Síntomas:**
```
ImportError: Missing optional dependency 'openpyxl'
```

**Solución:**
```bash
pip install openpyxl
```

### Problema 3: Conflictos de versiones

**Solución:**
```bash
# Desinstalar todas las librerías
pip freeze | xargs pip uninstall -y

# Reinstalar desde requirements.txt
pip install -r requirements.txt
```

### Problema 4: Streamlit no abre el navegador

**Solución:**
```bash
# Abrir manualmente
streamlit run model_monitoring.py --server.headless true
```

Luego visita: `http://localhost:8501`

### Problema 5: Puerto 8000 ya en uso (FastAPI)

**Solución:**
```bash
# Usar otro puerto
uvicorn model_deploy:app --port 8001
```

---

## 📚 Recursos Adicionales

### Documentación oficial:

| Librería | Documentación |
|----------|---------------|
| NumPy | https://numpy.org/doc/ |
| Pandas | https://pandas.pydata.org/docs/ |
| Scikit-learn | https://scikit-learn.org/stable/ |
| XGBoost | https://xgboost.readthedocs.io/ |
| FastAPI | https://fastapi.tiangolo.com/ |
| Streamlit | https://docs.streamlit.io/ |
| Evidently | https://docs.evidentlyai.com/ |
| Plotly | https://plotly.com/python/ |

### Tutoriales recomendados:
- **Scikit-learn:** https://scikit-learn.org/stable/tutorial/
- **FastAPI:** https://fastapi.tiangolo.com/tutorial/
- **Streamlit:** https://docs.streamlit.io/library/get-started

---

## ⚙️ Configuración Avanzada

### Optimización para producción

1. **Usar un gestor de dependencias más robusto:**
```bash
pip install poetry
poetry init
```

2. **Congelar versiones exactas:**
```bash
pip freeze > requirements-lock.txt
```

3. **Usar un gestor de versiones de Python:**
```bash
# pyenv (recomendado)
pyenv install 3.10.12
pyenv local 3.10.12
```

---

## 🐳 Containerización (Opcional)

### Dockerfile de ejemplo:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Construir y ejecutar:
```bash
docker build -t prediccion-pagos .
docker run -p 8000:8000 prediccion-pagos
```

---

## 📊 Comparación de Alternativas

| Necesidad | Librería Usada | Alternativas |
|-----------|----------------|--------------|
| DataFrames | Pandas | Polars, Dask |
| ML básico | Scikit-learn | Statsmodels, mljar |
| Boosting | XGBoost | LightGBM, CatBoost |
| API REST | FastAPI | Flask, Django REST |
| Dashboard | Streamlit | Dash, Gradio |
| Visualización | Plotly | Bokeh, Altair |

---

## ✅ Checklist de Instalación

Marca cada paso completado:

- [ ] Python 3.8-3.11 instalado
- [ ] Entorno virtual creado y activado
- [ ] `pip` actualizado a última versión
- [ ] `requirements.txt` creado
- [ ] Todas las librerías instaladas sin errores
- [ ] Importaciones verificadas (`python -c "import pandas, xgboost, fastapi"`)
- [ ] Jupyter notebook funcional (si aplica)

---

## 📞 Soporte

Si encuentras problemas:
1. Verifica la versión de Python: `python --version`
2. Verifica las versiones instaladas: `pip list`
3. Consulta los logs de error completos
4. Busca en Stack Overflow o la documentación oficial

---

**Última actualización:** Enero 2026  
**Versión del documento:** 1.0

---

**¡Instalación exitosa! 🎉 Ahora estás listo para ejecutar el proyecto.**

