# 📊 Reporte del Proyecto: Sistema de Predicción de Pagos a Tiempo

## 🎯 Objetivo del Proyecto

Desarrollar un sistema completo de Machine Learning para **predecir si un cliente pagará a tiempo** sus obligaciones financieras, permitiendo a la institución tomar decisiones proactivas sobre gestión de riesgos y políticas de crédito.

---

## 🏗️ Arquitectura del Sistema

El proyecto está estructurado en **5 módulos principales** que siguen el ciclo de vida completo de un proyecto de ML:

```
┌─────────────────────────────────────────────────────────┐
│                    1. CARGA DE DATOS                    │
│                   (cargar_datos.py)                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              2. FEATURE ENGINEERING                     │
│                 (ft_engineering.py)                     │
│   • Imputación de valores faltantes                    │
│   • Encoding de variables categóricas                  │
│   • Pipeline de preprocesamiento                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         3. ENTRENAMIENTO Y EVALUACIÓN                   │
│          (model_training_evaluation.py)                 │
│   • 8 algoritmos de ML diferentes                      │
│   • Validación cruzada                                 │
│   • Métricas de desempeño                              │
│   • Selección del mejor modelo (XGBoost)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                4. DESPLIEGUE (API)                      │
│                (model_deploy.py)                        │
│   • API REST con FastAPI                               │
│   • Endpoint /predict                                  │
│   • Predicciones en batch                              │
│   • Documentación automática                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             5. MONITOREO EN PRODUCCIÓN                  │
│               (model_monitoring.py)                     │
│   • Dashboard interactivo (Streamlit)                  │
│   • Detección de data drift                            │
│   • Métricas en tiempo real                            │
│   • Alertas automáticas                                │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos

| Archivo | Descripción |
|---------|-------------|
| `cargar_datos.py` | Carga datos desde archivos Excel |
| `ft_engineering.py` | Preprocesamiento y transformación de features |
| `model_training_evaluation.py` | Entrenamiento y evaluación de 8 modelos ML |
| `model_deploy.py` | API REST para servir predicciones |
| `model_monitoring.py` | Dashboard de monitoreo con Streamlit |
| `comprension_eda.ipynb` | Análisis exploratorio de datos |
| `Base_de_datos.xlsx` | Datos originales |
| `base_de_datoslimpia.csv` | Datos procesados |

---

## 🔧 Módulos Detallados

### 1️⃣ Carga de Datos (`cargar_datos.py`)

**Propósito:** Importar datos desde archivos Excel de forma robusta

**Características clave:**
- Manejo de rutas relativas y absolutas
- Validación de existencia de archivos
- Manejo de errores informativo

### 2️⃣ Feature Engineering (`ft_engineering.py`)

**Propósito:** Transformar datos crudos en features listos para ML

**Pipeline de preprocesamiento:**

```
FEATURES NUMÉRICAS          FEATURES CATEGÓRICAS
      ↓                            ↓
  Imputación (media)          Conversión a string
      ↓                            ↓
  Escalado estándar           Imputación ('missing')
                                   ↓
                              One-Hot Encoding
      ↓                            ↓
      └──────────┬─────────────────┘
                 ↓
         DATOS PROCESADOS
```

**Ventajas del pipeline:**
- Reproducibilidad garantizada
- Evita data leakage
- Fácil aplicación en producción

### 3️⃣ Entrenamiento y Evaluación (`model_training_evaluation.py`)

**Propósito:** Entrenar múltiples modelos y seleccionar el mejor

**Modelos evaluados:**
1. Regresión Logística
2. Linear SVC
3. SGD Classifier
4. Gaussian Naive Bayes
5. Linear Discriminant Analysis
6. Decision Tree
7. Random Forest
8. **XGBoost** ⭐ (modelo seleccionado)

**Métricas utilizadas:**
- **Accuracy:** Precisión general
- **Precision:** Cuántos de los predichos positivos son correctos
- **Recall:** Cuántos de los reales positivos capturamos (¡MÉTRICA CLAVE!)
- **F1-Score:** Balance entre precision y recall
- **ROC-AUC:** Capacidad discriminatoria del modelo

**¿Por qué XGBoost?**
- Alto recall para detectar clientes riesgosos
- Robusto ante desbalanceo de clases
- Maneja interacciones complejas entre variables

### 4️⃣ Despliegue (`model_deploy.py`)

**Propósito:** Servir predicciones en tiempo real mediante API REST

**Endpoints principales:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Página de inicio |
| `/docs` | GET | Documentación interactiva (Swagger) |
| `/health` | GET | Estado del servicio |
| `/predict` | POST | Realizar predicciones en batch |

**Ejemplo de uso:**

```python
# Solicitud POST a /predict
{
  "data": [
    {
      "edad": 35,
      "ingreso": 50000,
      "deuda": 10000,
      ...
    }
  ]
}

# Respuesta
{
  "predictions": [1],  # 1 = Pagará a tiempo
  "probabilities": [0.85],
  "threshold": 0.5
}
```

### 5️⃣ Monitoreo (`model_monitoring.py`)

**Propósito:** Supervisar el desempeño del modelo en producción

**Funcionalidades:**

1. **Distribución de predicciones**
   - Visualiza el balance de clases predichas
   - Detecta sesgos en las predicciones

2. **Evolución temporal**
   - Gráfica de predicciones a lo largo del tiempo
   - Identifica tendencias

3. **Data Drift Detection**
   - Compara distribución actual vs. datos de entrenamiento
   - Alerta cuando los datos cambian significativamente

4. **Métricas de negocio**
   - Tasa de predicciones positivas
   - Volumen de predicciones

---

## 📊 Resultados Esperados

### Métricas del modelo XGBoost (típicas):

| Métrica | Valor esperado | Interpretación |
|---------|----------------|----------------|
| Accuracy | ~85% | 85 de cada 100 predicciones son correctas |
| Recall | ~80-90% | Capturamos la mayoría de clientes riesgosos |
| Precision | ~75-85% | Pocas falsas alarmas |
| ROC-AUC | ~0.85-0.90 | Excelente capacidad discriminatoria |

---

## 🚀 Cómo Ejecutar el Proyecto

### Paso 1: Entrenamiento del modelo

```bash
python model_training_evaluation.py
```
Salida: `xgboost_modelo.json` (modelo entrenado)

### Paso 2: Desplegar la API

```bash
python model_deploy.py
```
API disponible en: `http://localhost:8000`

### Paso 3: Iniciar el dashboard de monitoreo

```bash
streamlit run model_monitoring.py
```
Dashboard disponible en: `http://localhost:8501`

---

## 🎓 Decisiones de Diseño y Buenas Prácticas

### ✅ Implementadas:

1. **Modularidad:** Cada componente es independiente y reutilizable
2. **Documentación exhaustiva:** Todos los módulos están bien documentados
3. **Manejo de errores:** Try-except en puntos críticos
4. **Validación de datos:** Pydantic valida entradas en la API
5. **Logging:** Registro de operaciones importantes
6. **Pipeline de preprocesamiento:** Evita data leakage
7. **Validación cruzada:** Evaluación robusta del modelo
8. **Monitoreo continuo:** Detección temprana de degradación

### 🎯 Métricas de negocio priorizadas:

**Recall > Precision** porque:
- Es más costoso NO identificar un cliente riesgoso (perder dinero)
- Que rechazar un buen cliente (costo de oportunidad menor)

---

## 🔮 Mejoras Futuras Sugeridas

1. **Automatización:**
   - Pipeline CI/CD para re-entrenamiento automático
   - Programar actualizaciones del modelo

2. **Escalabilidad:**
   - Dockerizar la aplicación
   - Desplegar en la nube (AWS, GCP, Azure)

3. **Características adicionales:**
   - Explicabilidad con SHAP values
   - A/B testing de modelos
   - Feedback loop (reentrenamiento con nuevos datos)

4. **Seguridad:**
   - Autenticación JWT en la API
   - Encriptación de datos sensibles
   - Rate limiting

---

## 👥 Equipo y Contacto

**Desarrollado por:** Santiago Joaquin Mozo 
**Fecha:** Enero 2026  
**Versión:** 1.0

---

## 📝 Conclusión

Este proyecto implementa un **sistema end-to-end de Machine Learning** que cubre todas las etapas del ciclo de vida de un modelo predictivo: desde la carga y preprocesamiento de datos, pasando por el entrenamiento y evaluación, hasta el despliegue y monitoreo en producción.

La arquitectura modular y las buenas prácticas implementadas aseguran que el sistema sea:
- ✅ **Mantenible:** Código limpio y bien documentado
- ✅ **Escalable:** Preparado para crecer
- ✅ **Confiable:** Con validaciones y monitoreo
- ✅ **Reproducible:** Pipelines estandarizados

Este proyecto demuestra competencias en:
- Machine Learning (scikit-learn, XGBoost)
- Ingeniería de software (FastAPI, pipelines)
- MLOps (despliegue, monitoreo, drift detection)
- Visualización de datos (Streamlit, Plotly)

---

**¡Proyecto completado exitosamente! 🎉**

