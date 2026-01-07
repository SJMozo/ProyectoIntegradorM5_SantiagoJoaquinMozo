"""
==============================================================================
API DE DESPLIEGUE DE MODELO DE PREDICCION DE PAGOS A TIEMPO
==============================================================================

CONTEXTO DEL NEGOCIO:
--------------------
Esta API REST despliega un modelo de Machine Learning (XGBoost) en produccion
para predecir si un cliente pagara a tiempo sus obligaciones crediticias.
El sistema esta disenado para procesar solicitudes en batch (lotes), permitiendo
evaluaciones rapidas y eficientes de multiples clientes simultaneamente.

ARQUITECTURA DE LA SOLUCION:
---------------------------
La API sigue una arquitectura RESTful implementada con FastAPI, que proporciona:

1. VALIDACION AUTOMATICA: Pydantic valida que los datos de entrada sean correctos
2. DOCUMENTACION INTERACTIVA: Swagger UI automatico en /docs
3. ALTO RENDIMIENTO: FastAPI es uno de los frameworks mas rapidos de Python
4. ESCALABILIDAD: Preparado para deployment en contenedores (Docker/Kubernetes)
5. MONITOREO: Endpoints adicionales para health checks y metricas

FLUJO DE PREDICCION:
-------------------
Cliente -> HTTP POST /predict -> Validacion (Pydantic) -> Preprocesamiento ->
Modelo XGBoost -> Probabilidades -> Threshold -> Clases binarias -> Respuesta JSON

METRICAS DE NEGOCIO:
-------------------
- Throughput: Predicciones procesadas por segundo
- Latencia: Tiempo de respuesta por solicitud
- Precision: Exactitud de las predicciones en produccion
- Disponibilidad: Uptime del servicio (objetivo: >99.9%)

DECISION DE THRESHOLD:
---------------------
Threshold actual: 0.5 (50%)
- Probabilidad >= 0.5 -> Clase 1 (Pagara a tiempo)
- Probabilidad < 0.5 -> Clase 0 (No pagara a tiempo)

Este threshold es ajustable segun el apetito de riesgo del negocio:
- Threshold mas alto (0.7): Mas conservador, menos aprobaciones
- Threshold mas bajo (0.3): Mas permisivo, mas aprobaciones pero mayor riesgo

SEGURIDAD Y MEJORES PRACTICAS:
-----------------------------
- Validacion de entrada con Pydantic (previene inyecciones)
- Manejo robusto de excepciones
- Logging de operaciones criticas
- Health checks para monitoreo
- Versionado del API

Autor: Santiago Joaquin Mozo
Version: 2.0
Fecha: 2026
==============================================================================
"""

# ==============================================================================
# 1. IMPORTACION DE LIBRERIAS
# ==============================================================================

# --- Framework web y validacion ---
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

# --- Machine Learning y procesamiento de datos ---
import pandas as pd
import numpy as np
import xgboost as xgb

# --- Servidor ASGI ---
import uvicorn

# --- Sistema y utilidades ---
import logging
import os
from datetime import datetime
import warnings

# Suprimir warnings para produccion
warnings.filterwarnings('ignore')


# ==============================================================================
# 2. CONFIGURACION DE LOGGING
# ==============================================================================

# Configurar logging para rastrear operaciones y errores
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_predictions.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 3. CONSTANTES Y CONFIGURACION
# ==============================================================================

# Ruta al modelo entrenado
MODEL_PATH = "xgb_model.json"

# Threshold de decision (ajustable segun politica de riesgo)
PREDICTION_THRESHOLD = 0.5

# Configuracion del servidor
API_HOST = "0.0.0.0"
API_PORT = 8000

# Limites de seguridad
MAX_BATCH_SIZE = 1000  # Maximo de registros por solicitud


# ==============================================================================
# 4. MODELOS DE DATOS CON PYDANTIC
# ==============================================================================

class PredictionInput(BaseModel):
    """
    Modelo de datos para una unica entrada de prediccion.
    
    PROPOSITO:
    ---------
    Define la estructura y validaciones de los datos que el modelo necesita
    para realizar una prediccion. Cada campo corresponde a una feature que
    el modelo XGBoost utilizo durante el entrenamiento.
    
    VALIDACION AUTOMATICA:
    ---------------------
    Pydantic valida automaticamente:
    - Tipos de datos correctos (float para todas las features)
    - Valores dentro de rangos esperados (via validators)
    - Campos requeridos presentes
    
    Si la validacion falla, la API retorna un error 422 con detalles
    especificos sobre que campo tiene problemas.
    
    FEATURES DEL MODELO:
    -------------------
    Las features incluyen:
    - Variables categoricas codificadas (bin_encoder, poly_ohe)
    - Variables numericas originales (capital, edad, salario)
    - Variables de comportamiento crediticio (puntaje, mora)
    - Variables de endeudamiento (creditos vigentes, sector)
    """
    
    # Features categoricas codificadas
    bin_encoder_tipo_laboral: float = Field(
        ...,
        description="Tipo de empleo codificado (0=Empleado, 1=Independiente)",
        ge=0.0,
        le=1.0
    )
    
    poly_ohe_tipo_credito_9: float = Field(
        ...,
        description="Indicador binario: Tipo de credito 9",
        ge=0.0,
        le=1.0
    )
    
    poly_ohe_tipo_credito_10: float = Field(
        ...,
        description="Indicador binario: Tipo de credito 10",
        ge=0.0,
        le=1.0
    )
    
    poly_ohe_tendencia_ingresos_Decreciente: float = Field(
        ...,
        description="Indicador binario: Tendencia de ingresos decreciente",
        ge=0.0,
        le=1.0
    )
    
    poly_ohe_tendencia_ingresos_Estable: float = Field(
        ...,
        description="Indicador binario: Tendencia de ingresos estable",
        ge=0.0,
        le=1.0
    )
    
    # Features numericas principales
    capital_prestado: float = Field(
        ...,
        description="Monto del prestamo solicitado",
        gt=0.0
    )
    
    plazo_meses: float = Field(
        ...,
        description="Plazo del prestamo en meses",
        gt=0.0,
        le=360.0  # Maximo 30 anos
    )
    
    edad_cliente: float = Field(
        ...,
        description="Edad del cliente en anos",
        ge=18.0,
        le=100.0
    )
    
    salario_cliente: float = Field(
        ...,
        description="Salario mensual del cliente",
        ge=0.0
    )
    
    total_otros_prestamos: float = Field(
        ...,
        description="Suma total de otros prestamos activos",
        ge=0.0
    )
    
    # Features de comportamiento crediticio
    puntaje_datacredito: float = Field(
        ...,
        description="Score de Datacredito del cliente",
        ge=0.0,
        le=1000.0
    )
    
    cant_creditosvigentes: float = Field(
        ...,
        description="Cantidad de creditos activos simultaneos",
        ge=0.0
    )
    
    huella_consulta: float = Field(
        ...,
        description="Numero de consultas recientes al historial crediticio",
        ge=0.0
    )
    
    # Features de estado financiero
    saldo_total: float = Field(
        ...,
        description="Saldo total de la deuda",
        ge=0.0
    )
    
    saldo_mora_codeudor: float = Field(
        ...,
        description="Saldo en mora del codeudor (si aplica)",
        ge=0.0
    )
    
    # Features por sector crediticio
    creditos_sectorCooperativo: float = Field(
        ...,
        description="Cantidad de creditos en sector cooperativo",
        ge=0.0
    )
    
    creditos_sectorReal: float = Field(
        ...,
        description="Cantidad de creditos en sector real/comercial",
        ge=0.0
    )
    
    class Config:
        """Configuracion adicional del modelo Pydantic"""
        schema_extra = {
            "example": {
                "bin_encoder_tipo_laboral": 1.0,
                "poly_ohe_tipo_credito_9": 0.0,
                "poly_ohe_tipo_credito_10": 1.0,
                "poly_ohe_tendencia_ingresos_Decreciente": 0.0,
                "poly_ohe_tendencia_ingresos_Estable": 1.0,
                "capital_prestado": 5000000.0,
                "plazo_meses": 36.0,
                "edad_cliente": 35.0,
                "salario_cliente": 3000000.0,
                "total_otros_prestamos": 2000000.0,
                "puntaje_datacredito": 650.0,
                "cant_creditosvigentes": 2.0,
                "huella_consulta": 3.0,
                "saldo_total": 1500000.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorCooperativo": 1.0,
                "creditos_sectorReal": 0.0
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Modelo de datos para solicitudes de prediccion en batch (lotes).
    
    PROPOSITO:
    ---------
    Permite procesar multiples predicciones en una sola solicitud HTTP,
    mejorando significativamente la eficiencia cuando se necesita evaluar
    muchos clientes simultaneamente.
    
    VENTAJAS DEL BATCH PROCESSING:
    ------------------------------
    1. EFICIENCIA: Una sola conexion HTTP para N predicciones
    2. PERFORMANCE: El overhead de red se amortiza entre todas las predicciones
    3. THROUGHPUT: Permite procesar miles de clientes rapidamente
    4. RECURSOS: Mejor utilizacion de CPU/GPU del servidor
    
    CASOS DE USO:
    ------------
    - Evaluacion de cartera completa de solicitudes pendientes
    - Scoring batch de clientes para campanas de marketing
    - Re-scoring periodico de cartera existente
    - Analisis de escenarios (what-if analysis)
    """
    
    batch: List[List[float]] = Field(
        ...,
        description="Lista de listas con los valores de las features en el orden correcto",
        min_items=1,
        max_items=MAX_BATCH_SIZE
    )
    
    feature_names: Optional[List[str]] = Field(
        None,
        description="Nombres de las features (opcional, para validacion)"
    )
    
    @validator('batch')
    def validate_batch_size(cls, v):
        """Valida que el batch no exceda el limite de seguridad"""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"El batch excede el limite maximo de {MAX_BATCH_SIZE} registros. "
                f"Para procesar mas datos, dividir en multiples solicitudes."
            )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "batch": [
                    [1.0, 0.0, 1.0, 0.0, 1.0, 5000000.0, 36.0, 35.0, 3000000.0, 
                     2000000.0, 650.0, 2.0, 3.0, 1500000.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 3000000.0, 24.0, 28.0, 2000000.0,
                     1000000.0, 700.0, 1.0, 2.0, 800000.0, 0.0, 0.0, 1.0]
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Modelo de respuesta con las predicciones y metadata.
    
    Incluye:
    - predictions: Lista de predicciones binarias (0 o 1)
    - probabilities: Probabilidades continuas del modelo
    - threshold_used: Threshold aplicado para la decision
    - timestamp: Momento de la prediccion
    - model_version: Version del modelo utilizado
    """
    
    predictions: List[int] = Field(..., description="Predicciones finales (0 o 1)")
    probabilities: List[float] = Field(..., description="Probabilidades del modelo")
    threshold_used: float = Field(..., description="Threshold de decision aplicado")
    timestamp: str = Field(..., description="Timestamp de la prediccion")
    batch_size: int = Field(..., description="Numero de registros procesados")
    model_version: str = Field(default="1.0", description="Version del modelo")


# ==============================================================================
# 5. INICIALIZACION DE LA APLICACION FASTAPI
# ==============================================================================

app = FastAPI(
    title="API de Prediccion de Riesgo Crediticio",
    description="""
    ## API REST para prediccion de comportamiento de pago
    
    Esta API despliega un modelo XGBoost entrenado para predecir si un cliente
    pagara a tiempo sus obligaciones crediticias.
    
    ### Endpoints Disponibles:
    
    - **GET /**: Health check y mensaje de bienvenida
    - **GET /health**: Estado detallado del sistema
    - **POST /predict_batch**: Predicciones en batch (lote)
    - **GET /model_info**: Informacion sobre el modelo desplegado
    
    ### Documentacion Interactiva:
    
    - **Swagger UI**: /docs (esta pagina)
    - **ReDoc**: /redoc (documentacion alternativa)
    
    ### Como usar la API:
    
    1. Preparar los datos de entrada en el formato requerido
    2. Enviar POST a /predict_batch con el batch de datos
    3. Recibir predicciones y probabilidades en JSON
    4. Interpretar resultados segun el threshold configurado
    
    ### Politica de Threshold:
    
    El threshold actual es 0.5 (50%). Esto significa:
    - Probabilidad >= 0.5: Cliente clasificado como "Pagara a tiempo" (1)
    - Probabilidad < 0.5: Cliente clasificado como "NO pagara a tiempo" (0)
    
    Este threshold puede ajustarse segun la tolerancia al riesgo del negocio.
    """,
    version="2.0.0",
    contact={
        "name": "Equipo de Data Science",
        "email": "analytics@empresa.com"
    },
    license_info={
        "name": "Proprietary"
    }
)


# ==============================================================================
# 6. CARGA DEL MODELO AL INICIAR LA APLICACION
# ==============================================================================

# Variable global para almacenar el modelo cargado
model = None
model_features = None
model_load_time = None

def load_model():
    """
    Carga el modelo XGBoost entrenado desde disco.
    
    ESTRATEGIA DE CARGA:
    -------------------
    El modelo se carga UNA SOLA VEZ al iniciar la aplicacion y se mantiene
    en memoria para todas las solicitudes subsecuentes. Esto es critico para
    el rendimiento en produccion.
    
    ALTERNATIVAS DE DEPLOYMENT:
    --------------------------
    1. Carga en memoria (actual): Rapido, pero consume RAM
    2. Carga bajo demanda: Ahorra RAM, pero lento
    3. Carga con cache: Hibrido, complejidad media
    
    La opcion 1 es optima para APIs de alta demanda donde la latencia
    es critica y hay RAM disponible.
    
    Returns:
    -------
    tuple: (modelo, nombres_features, tiempo_carga)
    
    Raises:
    ------
    FileNotFoundError: Si el archivo del modelo no existe
    Exception: Si hay error al cargar el modelo
    """
    global model, model_features, model_load_time
    
    try:
        logger.info(f"Iniciando carga del modelo desde: {MODEL_PATH}")
        
        # Verificar que el archivo existe
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Archivo de modelo no encontrado: {MODEL_PATH}\n"
                f"Asegurate de que el modelo fue entrenado y guardado correctamente."
            )
        
        # Cargar modelo XGBoost
        start_time = datetime.now()
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        load_duration = (datetime.now() - start_time).total_seconds()
        
        # Extraer nombres de features del modelo
        model_features = model.feature_names
        model_load_time = datetime.now().isoformat()
        
        logger.info(f"Modelo cargado exitosamente en {load_duration:.3f} segundos")
        logger.info(f"Features del modelo: {len(model_features)}")
        logger.info(f"Nombres de features: {model_features}")
        
        return model, model_features, model_load_time
        
    except FileNotFoundError as e:
        logger.error(f"Error: Archivo de modelo no encontrado - {e}")
        raise
        
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise

# Cargar modelo al iniciar la aplicacion
try:
    load_model()
    logger.info("API inicializada correctamente con modelo cargado")
except Exception as e:
    logger.critical(f"FALLO CRITICO: No se pudo cargar el modelo - {e}")
    logger.critical("La API se iniciara pero NO podra realizar predicciones")
    model = None


# ==============================================================================
# 7. ENDPOINTS DE LA API
# ==============================================================================

@app.get("/", tags=["General"])
def root():
    """
    Endpoint raiz: Health check basico y mensaje de bienvenida.
    
    PROPOSITO:
    ---------
    Proporciona una forma rapida de verificar que la API esta activa y
    funcionando. Util para:
    - Monitoreo automatizado (health checks)
    - Verificacion manual rapida
    - Load balancers (verificar que el servicio responde)
    
    Returns:
    -------
    dict: Mensaje de bienvenida y estado basico
    """
    return {
        "message": "API de Prediccion de Riesgo Crediticio",
        "status": "activa",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict_batch",
            "info": "/model_info",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    """
    Health check detallado del sistema.
    
    PROPOSITO:
    ---------
    Proporciona informacion detallada sobre el estado del servicio,
    incluyendo:
    - Estado del modelo (cargado/no cargado)
    - Tiempo de carga
    - Informacion del sistema
    - Estadisticas de uso (futuro)
    
    MONITOREO EN PRODUCCION:
    -----------------------
    Este endpoint debe ser monitoreado por sistemas como:
    - Kubernetes liveness/readiness probes
    - Herramientas de monitoreo (Prometheus, Datadog)
    - Sistemas de alertas
    
    Returns:
    -------
    dict: Estado detallado del sistema
    """
    return {
        "status": "healthy" if model is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model is not None,
            "load_time": model_load_time,
            "features_count": len(model_features) if model_features else 0,
            "path": MODEL_PATH
        },
        "configuration": {
            "threshold": PREDICTION_THRESHOLD,
            "max_batch_size": MAX_BATCH_SIZE,
            "host": API_HOST,
            "port": API_PORT
        }
    }


@app.get("/model_info", tags=["Model"])
def model_info():
    """
    Informacion detallada sobre el modelo desplegado.
    
    TRANSPARENCIA DEL MODELO:
    ------------------------
    Proporciona metadata sobre el modelo para:
    - Auditoria y compliance
    - Debugging de problemas en produccion
    - Documentacion para usuarios de la API
    - Validacion de que se esta usando el modelo correcto
    
    Returns:
    -------
    dict: Informacion del modelo
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. El servicio no puede realizar predicciones."
        )
    
    return {
        "model_type": "XGBoost Booster",
        "model_path": MODEL_PATH,
        "load_timestamp": model_load_time,
        "features": {
            "count": len(model_features),
            "names": model_features
        },
        "configuration": {
            "threshold": PREDICTION_THRESHOLD,
            "threshold_interpretation": {
                ">=0.5": "Pagara a tiempo (1)",
                "<0.5": "NO pagara a tiempo (0)"
            }
        },
        "input_format": "Batch de listas de floats en orden de features",
        "output_format": "Lista de predicciones binarias (0 o 1)"
    }


@app.post("/predict_batch", response_model=PredictionResponse, tags=["Predictions"])
def predict_batch(input_data: BatchPredictionInput):
    """
    Endpoint principal: Predicciones en batch (lotes).
    
    FLUJO DE PROCESAMIENTO:
    ----------------------
    1. VALIDACION: Pydantic valida el formato de entrada
    2. CONVERSION: Datos se convierten a DataFrame de pandas
    3. ALINEACION: Columnas se ordenan segun features del modelo
    4. TRANSFORMACION: DataFrame se convierte a DMatrix de XGBoost
    5. PREDICCION: Modelo genera probabilidades
    6. THRESHOLD: Probabilidades se convierten a clases binarias
    7. RESPUESTA: JSON con predicciones y metadata
    
    MANEJO DE ERRORES:
    -----------------
    - 422: Validacion falla (formato incorrecto)
    - 503: Modelo no disponible
    - 500: Error interno al procesar
    
    Args:
        input_data: Batch de datos validados por Pydantic
    
    Returns:
        PredictionResponse: Predicciones, probabilidades y metadata
    
    Raises:
        HTTPException: Si hay error en el procesamiento
    """
    # Verificar que el modelo esta disponible
    if model is None:
        logger.error("Intento de prediccion con modelo no cargado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Modelo no disponible",
                "message": "El modelo no pudo ser cargado. Contacta al administrador del sistema.",
                "suggestion": "Verifica que el archivo del modelo exista y sea valido"
            }
        )
    
    try:
        logger.info(f"Iniciando prediccion batch de {len(input_data.batch)} registros")
        
        # PASO 1: Convertir batch a DataFrame
        # Cada sublista es un registro, cada elemento es una feature
        df = pd.DataFrame(input_data.batch, columns=model_features)
        
        logger.debug(f"DataFrame creado: {df.shape}")
        
        # PASO 2: Validar que todas las features esten presentes
        missing_features = set(model_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Features faltantes en los datos de entrada: {missing_features}"
            )
        
        # PASO 3: Asegurar orden correcto de columnas
        df = df[model_features]
        
        # PASO 4: Convertir a DMatrix (formato nativo de XGBoost)
        # DMatrix es mas eficiente que DataFrames para XGBoost
        dmatrix = xgb.DMatrix(df)
        
        logger.debug("DMatrix creada exitosamente")
        
        # PASO 5: Generar predicciones (probabilidades)
        # El modelo retorna probabilidades continuas [0, 1]
        predictions_proba = model.predict(dmatrix)
        
        logger.debug(f"Probabilidades generadas: {len(predictions_proba)}")
        
        # PASO 6: Aplicar threshold para convertir a clases binarias
        # Esta es una decision de negocio, no tecnica
        final_predictions = [
            1 if prob >= PREDICTION_THRESHOLD else 0 
            for prob in predictions_proba
        ]
        
        logger.info(
            f"Prediccion completada: {sum(final_predictions)} predichos como '1' (Pagara), "
            f"{len(final_predictions) - sum(final_predictions)} predichos como '0' (No pagara)"
        )
        
        # PASO 7: Construir respuesta estructurada
        response = PredictionResponse(
            predictions=final_predictions,
            probabilities=predictions_proba.tolist(),
            threshold_used=PREDICTION_THRESHOLD,
            timestamp=datetime.now().isoformat(),
            batch_size=len(final_predictions),
            model_version="1.0"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Error de validacion: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Error de validacion",
                "message": str(e),
                "suggestion": "Verifica que los datos tengan el formato correcto"
            }
        )
        
    except Exception as e:
        logger.error(f"Error inesperado al procesar prediccion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Error interno del servidor",
                "message": "Ocurrio un error al procesar la prediccion",
                "trace": str(e)
            }
        )


# ==============================================================================
# 8. PUNTO DE ENTRADA PARA EJECUCION DIRECTA
# ==============================================================================

if __name__ == '__main__':
    """
    Bloque de ejecucion principal.
    
    PROPOSITO:
    ---------
    Permite ejecutar la API directamente con Python para desarrollo y pruebas
    locales. En produccion, se usaria un servidor ASGI como Gunicorn o
    directamente Uvicorn con configuracion optimizada.
    
    CONFIGURACION DE DESARROLLO:
    ---------------------------
    - reload=True: Recarga automatica al detectar cambios en el codigo
    - host="0.0.0.0": Acepta conexiones de cualquier IP (necesario para Docker)
    - port=8000: Puerto estandar (configurable)
    
    DEPLOYMENT EN PRODUCCION:
    ------------------------
    Para produccion, usar:
    ```
    uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --workers 4
    ```
    
    O con Gunicorn (mas robusto):
    ```
    gunicorn model_deploy:app -w 4 -k uvicorn.workers.UvicornWorker
    ```
    
    ESCALABILIDAD:
    -------------
    - Workers: Procesos paralelos (1 por core de CPU)
    - Threads: Hilos por worker (para I/O bound tasks)
    - Load Balancer: Nginx/HAProxy para distribuir carga
    - Containerizacion: Docker + Kubernetes para orquestacion
    """
    
    logger.info("Iniciando servidor de desarrollo...")
    logger.info(f"API disponible en: http://{API_HOST}:{API_PORT}")
    logger.info(f"Documentacion interactiva en: http://{API_HOST}:{API_PORT}/docs")
    logger.info("Presiona CTRL+C para detener el servidor")
    
    uvicorn.run(
        "model_deploy:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )


# ==============================================================================
# FIN DEL MODULO
# ==============================================================================

"""
INSTRUCCIONES DE USO:
--------------------

1. DESARROLLO LOCAL:
   python model_deploy.py
   
2. PRODUCCION:
   uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --workers 4
   
3. DOCKER:
   FROM python:3.9
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
   
4. PRUEBAS:
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/predict_batch -H "Content-Type: application/json" -d @test_data.json
   
5. DOCUMENTACION:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)
"""
