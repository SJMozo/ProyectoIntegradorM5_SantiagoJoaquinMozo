"""
==============================================================================
SISTEMA DE MONITOREO DE MODELOS EN PRODUCCIÓN
==============================================================================

CONTEXTO DEL NEGOCIO:
--------------------
Este dashboard interactivo permite monitorear en tiempo real el desempeño
de modelos de Machine Learning en producción. El monitoreo continuo es 
crítico para:

- Detectar degradación del modelo (model drift)
- Identificar cambios en los datos de entrada (data drift)
- Asegurar que las predicciones mantienen calidad esperada
- Tomar decisiones informadas sobre cuándo re-entrenar

PROBLEMÁTICA CLAVE:
------------------
Los modelos ML pueden degradarse con el tiempo porque:
1. Los patrones en los datos cambian (concept drift)
2. La distribución de features varía (data drift)
3. Nuevos escenarios no vistos en entrenamiento aparecen
4. Las relaciones entre variables evolucionan

ARQUITECTURA DEL SISTEMA:
------------------------
1. Carga de datos de referencia (baseline de entrenamiento)
2. Generación de predicciones vía API REST
3. Registro histórico de predicciones (logging)
4. Análisis de drift con Evidently AI
5. Visualizaciones interactivas con Streamlit
6. Alertas automáticas de anomalías

MÉTRICAS MONITOREADAS:
---------------------
- Distribución de predicciones (detectar sesgos)
- Evolución temporal (detectar tendencias)
- Data drift por feature (cambios en inputs)
- Tasa de predicciones positivas (cambios en outcomes)

Autor: Santiago Joaquin Mozo
Versión: 2.0
Fecha: 2026
==============================================================================
"""

# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================

# --- Librerías del sistema ---
import os
import time
from typing import Optional, Dict, List

# --- Análisis de datos ---
import pandas as pd
import numpy as np

# --- API y comunicación ---
import requests

# --- Visualización interactiva ---
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Monitoreo de ML ---
from evidently import Report
from evidently.presets import DataDriftPreset

# --- Machine Learning ---
from sklearn.model_selection import train_test_split


# ==============================================================================
# 2. CONFIGURACIÓN DEL DASHBOARD
# ==============================================================================

# Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="Monitoreo de Modelos ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 3. CONSTANTES Y PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================

# --- Rutas de archivos ---
DATA_PATH = "base_de_datoslimpia.csv"  # Datos limpios para referencia
MONITOR_LOG = "monitoring_log.csv"     # Log de predicciones en producción

# --- Configuración de API ---
API_URL = "http://localhost:8000/predict_batch"
API_TIMEOUT = 30  # segundos

# --- Parámetros de análisis ---
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% de datos como "nuevos" para simular producción
TARGET_COLUMN = "Pago_atiempo"

# --- Umbrales de alerta ---
DRIFT_THRESHOLD = 0.5  # Umbral para considerar drift significativo
PREDICTION_CHANGE_ALERT = 0.15  # 15% de cambio en distribución es alerta

# --- Configuración visual ---
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'reference': 'lightblue',
    'current': 'orange'
}


# ==============================================================================
# 4. FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

@st.cache_data
def load_reference_data():
    """
    Carga y prepara los datos de referencia para monitoreo.
    
    PROPÓSITO:
    ---------
    Los datos de referencia representan el "estado normal" del modelo
    durante el entrenamiento. Sirven como baseline para:
    - Comparar distribuciones de features
    - Detectar data drift
    - Evaluar si las predicciones actuales son consistentes
    
    ESTRATEGIA DE DIVISIÓN:
    ----------------------
    Dividimos los datos en:
    - X_ref, y_ref: Datos "históricos" (80%) - baseline de entrenamiento
    - X_new, y_new: Datos "nuevos" (20%) - simulan datos en producción
    
    Esta división permite simular un escenario realista donde tenemos
    datos históricos conocidos y datos nuevos a monitorear.
    
    Returns:
    -------
    tuple: (X_ref, X_new, y_ref, y_new, df_original)
        - X_ref: Features de referencia (baseline)
        - X_new: Features "nuevas" (simulan producción)
        - y_ref: Target de referencia
        - y_new: Target "nuevo"
        - df_original: DataFrame original completo
    
    Raises:
    ------
    FileNotFoundError: Si no se encuentra el archivo de datos
    """
    try:
        # Cargar datos limpios
        df = pd.read_csv(DATA_PATH)
        
        # Validar que existe la columna target
        if TARGET_COLUMN not in df.columns:
            st.error(f"Error: La columna '{TARGET_COLUMN}' no existe en el dataset")
            st.stop()
        
        # Separar features y target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        # División estratificada para mantener proporción de clases
        X_ref, X_new, y_ref, y_new = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y  # Mantiene balance de clases
        )
        
        return X_ref, X_new, y_ref, y_new, df
        
    except FileNotFoundError:
        st.error(f"**Error crítico**: No se encontró el archivo '{DATA_PATH}'")
        st.info("Verifica que el archivo exista en el directorio del proyecto")
        st.stop()
        
    except Exception as e:
        st.error(f"**Error al cargar datos**: {str(e)}")
        st.stop()


# ==============================================================================
# 5. FUNCIONES DE COMUNICACIÓN CON API
# ==============================================================================

def get_predictions(X_batch: pd.DataFrame) -> Optional[List[float]]:
    """
    Obtiene predicciones del modelo en producción vía API REST.
    
    ARQUITECTURA:
    ------------
    Este sistema asume que el modelo está desplegado como un servicio
    REST que acepta lotes de datos y retorna predicciones. Esta es
    la arquitectura estándar para ML en producción porque:
    
    - Desacopla el modelo del dashboard
    - Permite escalar independientemente
    - Facilita actualizaciones del modelo sin cambiar el monitoreo
    - Soporta múltiples clientes simultáneos
    
    Args:
        X_batch: DataFrame con features para predecir
        
    Returns:
        List[float] o None: Lista de predicciones o None si hay error
        
    Notes:
        - Las predicciones son probabilidades [0, 1]
        - None indica que el servicio no está disponible
        - Los errores se muestran al usuario vía Streamlit
    """
    # Preparar payload (convertir DataFrame a lista de listas)
    payload = {
        "batch": X_batch.values.tolist(),
        "feature_names": X_batch.columns.tolist()  # Para validación
    }
    
    try:
        # Realizar petición POST con timeout
        response = requests.post(
            API_URL,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        # Validar respuesta HTTP
        response.raise_for_status()
        
        # Extraer predicciones
        result = response.json()
        predictions = result.get("predictions")
        
        if predictions is None:
            st.warning("La API no retornó predicciones")
            return None
        
        # Validar cantidad de predicciones
        if len(predictions) != len(X_batch):
            st.warning(
                f"Inconsistencia: {len(X_batch)} muestras enviadas, "
                f"pero {len(predictions)} predicciones recibidas"
            )
        
        return predictions
        
    except requests.exceptions.Timeout:
        st.error(
            f"**Timeout**: El servidor no respondió en {API_TIMEOUT}s\n\n"
            "**Posibles causas:**\n"
            "- El servicio está sobrecargado\n"
            "- El modelo es muy lento\n"
            "- Problemas de red"
        )
        return None
        
    except requests.exceptions.ConnectionError:
        st.error(
            f"**Error de conexión**: No se pudo conectar a `{API_URL}`\n\n"
            "**Verifica que:**\n"
            "- El servicio API está corriendo\n"
            "- La URL es correcta\n"
            "- No hay firewall bloqueando la conexión"
        )
        return None
        
    except requests.exceptions.HTTPError as e:
        st.error(
            f"**Error HTTP {response.status_code}**: {e}\n\n"
            f"**Respuesta del servidor:**\n```\n{response.text}\n```"
        )
        return None
        
    except Exception as e:
        st.error(f"**Error inesperado**: {str(e)}")
        return None


# ==============================================================================
# 6. FUNCIONES DE LOGGING Y PERSISTENCIA
# ==============================================================================

def log_predictions(X_batch: pd.DataFrame, predictions: List[float]) -> bool:
    """
    Registra predicciones en log histórico para análisis posterior.
    
    IMPORTANCIA DEL LOGGING:
    -----------------------
    El registro de predicciones es crítico para:
    - Auditoría y trazabilidad
    - Análisis de tendencias temporales
    - Detección de anomalías
    - Evaluación retrospectiva (cuando se conocen valores reales)
    - Cumplimiento regulatorio (finanzas, salud)
    
    FORMATO DEL LOG:
    ---------------
    Cada registro incluye:
    - Todas las features de entrada
    - Predicción del modelo
    - Timestamp exacto
    
    Args:
        X_batch: Features utilizadas para predicción
        predictions: Predicciones retornadas por el modelo
        
    Returns:
        bool: True si el logging fue exitoso, False en caso contrario
    """
    try:
        # Crear DataFrame con features y predicciones
        log_df = X_batch.copy()
        log_df["prediction"] = predictions
        log_df["timestamp"] = pd.Timestamp.now()
        
        # Determinar modo de escritura
        if os.path.exists(MONITOR_LOG):
            # Append sin encabezado si el archivo existe
            log_df.to_csv(MONITOR_LOG, mode="a", header=False, index=False)
        else:
            # Crear nuevo archivo con encabezado
            log_df.to_csv(MONITOR_LOG, mode="w", header=True, index=False)
        
        return True
        
    except PermissionError:
        st.error(
            f"**Error de permisos**: No se puede escribir en `{MONITOR_LOG}`\n\n"
            "El archivo puede estar abierto en otra aplicación."
        )
        return False
        
    except Exception as e:
        st.error(f"**Error al guardar log**: {str(e)}")
        return False


def load_monitoring_log() -> Optional[pd.DataFrame]:
    """
    Carga el log histórico de monitoreo.
    
    Returns:
        DataFrame con el log o None si no existe
    """
    if not os.path.exists(MONITOR_LOG):
        return None
    
    try:
        df = pd.read_csv(MONITOR_LOG)
        
        # Convertir timestamp a datetime si existe
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar log: {str(e)}")
        return None


# ==============================================================================
# 7. FUNCIONES DE ANÁLISIS DE DRIFT
# ==============================================================================

def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Optional[Report]:
    """
    Genera reporte de Data Drift usando Evidently AI.
    
    ¿QUÉ ES DATA DRIFT?
    -------------------
    Data drift ocurre cuando la distribución de las features cambia
    entre entrenamiento y producción. Esto puede causar:
    
    - Degradación del desempeño del modelo
    - Predicciones menos confiables
    - Violación de asunciones del modelo
    
    TIPOS DE DRIFT:
    --------------
    1. **Covariate Drift**: Cambios en P(X) - distribución de features
    2. **Concept Drift**: Cambios en P(Y|X) - relación entre X e Y
    3. **Label Drift**: Cambios en P(Y) - distribución del target
    
    MÉTODOS DE DETECCIÓN:
    --------------------
    Evidently usa múltiples tests estadísticos:
    - Kolmogorov-Smirnov (variables continuas)
    - Chi-cuadrado (variables categóricas)
    - Population Stability Index (PSI)
    
    Args:
        reference_data: Datos de referencia (baseline)
        current_data: Datos actuales (producción)
        
    Returns:
        Report de Evidently o None si hay error
    """
    try:
        # Crear reporte con preset de Data Drift
        report = Report(metrics=[DataDriftPreset()])
        
        # Ejecutar análisis
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        return report
        
    except Exception as e:
        st.error(f"Error generando reporte de drift: {str(e)}")
        return None


def extract_drift_metrics(report: Report) -> Dict:
    """
    Extrae métricas clave del reporte de drift.
    
    Returns:
        Diccionario con métricas interpretables
    """
    try:
        drift_data = report.as_dict()
        
        if 'metrics' not in drift_data or len(drift_data['metrics']) == 0:
            return {
                'dataset_drift': False,
                'drifted_features': 0,
                'total_features': 0,
                'drift_percentage': 0.0
            }
        
        result = drift_data['metrics'][0].get('result', {})
        dataset_drift = result.get('dataset_drift', False)
        drift_by_columns = result.get('drift_by_columns', {})
        
        drifted_features = sum(1 for v in drift_by_columns.values() if v)
        total_features = len(drift_by_columns)
        drift_percentage = (drifted_features / total_features * 100) if total_features > 0 else 0
        
        return {
            'dataset_drift': dataset_drift,
            'drifted_features': drifted_features,
            'total_features': total_features,
            'drift_percentage': drift_percentage,
            'features_detail': drift_by_columns
        }
        
    except Exception as e:
        st.warning(f"No se pudieron extraer métricas de drift: {str(e)}")
        return {}


# ==============================================================================
# 8. CARGA INICIAL DE DATOS
# ==============================================================================

# Cargar datos de referencia (cacheado para eficiencia)
X_ref, X_new, y_ref, y_new, df_original = load_reference_data()


# ==============================================================================
# 9. INTERFAZ DE USUARIO: HEADER Y CONTEXTO
# ==============================================================================

# --- Header principal con narrativa ---
st.markdown("""
<div style='background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%); 
            padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>
        Dashboard de Monitoreo de Modelos ML
    </h1>
    <p style='color: white; margin-top: 0.5rem; font-size: 1.1rem;'>
        Supervisión en Tiempo Real del Sistema de Predicción de Riesgo de Pago
    </p>
</div>
""", unsafe_allow_html=True)

# --- Información contextual en expander ---
with st.expander("¿Qué es este dashboard y por qué es importante?", expanded=False):
    st.markdown("""
    ### Propósito del Sistema
    
    Este dashboard monitorea la **salud** de nuestro modelo de Machine Learning en producción.
    Los modelos ML pueden degradarse con el tiempo debido a cambios en:
    
    - **Datos de entrada** (los clientes cambian sus patrones)
    - **Ambiente de negocio** (crisis económicas, nuevas regulaciones)
    - **Contexto general** (tendencias del mercado, competencia)
    
    ### Métricas Monitoreadas
    
    1. **Distribución de Predicciones**: ¿Las predicciones siguen patrones esperados?
    2. **Data Drift**: ¿Los datos de entrada son similares al entrenamiento?
    3. **Tendencias Temporales**: ¿Hay cambios sistemáticos en el tiempo?
    4. **Tasa de Alertas**: ¿Estamos detectando más o menos casos riesgosos?
    
    ### Cuándo Preocuparse
    
    - **Data Drift Alto** (>50%): El modelo puede estar viendo datos muy diferentes
    - **Cambios Bruscos**: Saltos repentinos en distribución de predicciones
    - **Tendencias Sostenidas**: Incremento/decremento continuo sin explicación
    
    ### Acciones Recomendadas
    
    - Si hay drift significativo → **Re-entrenar modelo** con datos recientes
    - Si las predicciones son inconsistentes → **Investigar calidad de datos**
    - Si hay tendencias preocupantes → **Consultar con equipos de negocio**
    """)


# ==============================================================================
# 10. SIDEBAR: CONTROLES Y CONFIGURACIÓN
# ==============================================================================

st.sidebar.header("Configuración")

st.sidebar.markdown("""
---
### Datos de Referencia
Estos datos representan el comportamiento "normal" del modelo durante entrenamiento.
""")

st.sidebar.metric(
    "Muestras de Referencia",
    f"{len(X_ref):,}",
    help="Datos históricos usados como baseline"
)

st.sidebar.metric(
    "Muestras Nuevas Disponibles",
    f"{len(X_new):,}",
    help="Datos 'nuevos' para simular producción"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Generar Predicciones")

# Slider para tamaño de muestra
sample_size = st.sidebar.slider(
    "Tamaño de muestra:",
    min_value=50,
    max_value=min(500, len(X_new)),
    value=min(200, len(X_new)),
    step=50,
    help="Número de registros a enviar a la API para predicción"
)

# Botón de generación con narrativa
if st.sidebar.button("Generar Nuevas Predicciones", type="primary"):
    with st.spinner(f"Enviando {sample_size} registros a la API..."):
        # Muestrear datos
        sample = X_new.sample(n=sample_size, random_state=int(time.time()))
        
        # Obtener predicciones
        preds = get_predictions(sample)
        
        if preds:
            # Guardar en log
            success = log_predictions(sample, preds)
            
            if success:
                st.sidebar.success(
                    f"¡Éxito!\n\n"
                    f"- {len(preds)} predicciones generadas\n"
                    f"- Log actualizado correctamente\n"
                    f"- Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                # Recargar página para mostrar nuevos datos
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("Error al guardar predicciones")
        else:
            st.sidebar.error(
                "No se pudieron obtener predicciones\n\n"
                "Verifica que el servicio API esté corriendo."
            )

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
**Tip**: Genera predicciones periódicamente para monitorear 
la evolución temporal del modelo.
</small>
""", unsafe_allow_html=True)


# ==============================================================================
# 11. CONTENIDO PRINCIPAL: ANÁLISIS Y VISUALIZACIONES
# ==============================================================================

# Cargar log de monitoreo
logged_data = load_monitoring_log()

# Validar que hay datos disponibles
if logged_data is None:
    # --- Estado inicial: Sin datos ---
    st.info("""
    ### ¡Bienvenido al Sistema de Monitoreo!
    
    Aun no hay datos de monitoreo registrados. Para comenzar:
    
    1. Asegurate de que el servicio API este corriendo en `{API_URL}`
    2. Ajusta el tamano de muestra en el panel lateral (recomendado: 200)
    3. Presiona **"Generar Nuevas Predicciones"**
    4. Explora las visualizaciones y metricas que apareceran
    
    El sistema comenzara a recopilar datos automaticamente.
    """.format(API_URL=API_URL))
    
    st.stop()

if len(logged_data) == 0:
    # --- Sin datos en el log ---
    st.info("""
    ### El log de monitoreo esta vacio
    
    Para comenzar:
    
    1. Asegurate de que el servicio API este corriendo en `{API_URL}`
    2. Ajusta el tamano de muestra en el panel lateral (recomendado: 200)
    3. Presiona **"Generar Nuevas Predicciones"**
    4. Explora las visualizaciones y metricas que apareceran
    """.format(API_URL=API_URL))
    
    st.stop()

# ==============================================================================
# 12. MÉTRICAS PRINCIPALES (KPIs)
# ==============================================================================

st.markdown("### Indicadores Clave de Desempeño (KPIs)")

# Calcular estadísticas
total_predictions = len(logged_data)
mean_prediction = logged_data['prediction'].mean()
std_prediction = logged_data['prediction'].std()
positive_rate = (logged_data['prediction'] > 0.5).mean() * 100

# Calcular tasa de "no pago a tiempo" (clase 0)
risky_rate = (logged_data['prediction'] < 0.5).mean() * 100

# Mostrar KPIs en columnas
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Predicciones",
        f"{total_predictions:,}",
        help="Número total de predicciones registradas"
    )

with col2:
    st.metric(
        "Predicción Media",
        f"{mean_prediction:.3f}",
        help="Promedio de probabilidades predichas"
    )

with col3:
    st.metric(
        "Desviación Estándar",
        f"{std_prediction:.3f}",
        help="Variabilidad en las predicciones"
    )

with col4:
    # Delta comparado con datos de referencia si es posible
    st.metric(
        "Tasa 'Pago a Tiempo'",
        f"{positive_rate:.1f}%",
        help="% de clientes predichos como 'pagarán a tiempo'"
    )

with col5:
    st.metric(
        "Tasa 'Riesgo Alto'",
        f"{risky_rate:.1f}%",
        delta=f"{risky_rate - (100 - positive_rate):.1f}%",
        delta_color="inverse",
        help="% de clientes predichos como 'no pagarán a tiempo' (crítico)"
    )

# Interpretación automática de KPIs
st.markdown("---")
with st.expander("Interpretación de KPIs", expanded=False):
    interpretations = []
    
    # Análisis de tasa de riesgo
    if risky_rate > 15:
        interpretations.append(
            f"**Alta tasa de riesgo ({risky_rate:.1f}%)**: "
            "El modelo está identificando muchos casos potencialmente problemáticos. "
            "Esto podría indicar:\n"
            "   - Deterioro en la calidad de los clientes\n"
            "   - Cambios en el perfil de solicitantes\n"
            "   - El modelo puede estar siendo demasiado conservador"
        )
    elif risky_rate < 3:
        interpretations.append(
            f"**Baja tasa de riesgo ({risky_rate:.1f}%)**: "
            "El modelo predice que la mayoría de clientes pagarán a tiempo. "
            "Sin embargo, verifica que no esté siendo demasiado optimista."
        )
    else:
        interpretations.append(
            f"**Tasa de riesgo moderada ({risky_rate:.1f}%)**: "
            "El modelo mantiene un balance razonable en sus predicciones."
        )
    
    # Análisis de variabilidad
    if std_prediction < 0.1:
        interpretations.append(
            f"**Baja variabilidad (σ={std_prediction:.3f})**: "
            "Las predicciones son muy similares entre sí. Posibles causas:\n"
            "   - El modelo puede no estar discriminando bien entre casos\n"
            "   - Los datos de entrada son muy homogéneos\n"
            "   - Considerar revisar la confianza del modelo"
        )
    elif std_prediction > 0.4:
        interpretations.append(
            f"**Alta variabilidad (σ={std_prediction:.3f})**: "
            "Las predicciones cubren un rango amplio, lo cual es esperado si "
            "los clientes tienen perfiles diversos."
        )
    
    # Mostrar interpretaciones
    for interpretation in interpretations:
        st.markdown(interpretation)


# ==============================================================================
# 13. TABS DE ANÁLISIS DETALLADO
# ==============================================================================

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "Visualizaciones",
    "Análisis de Drift",
    "Log de Datos",
    "Insights Accionables"
])

# ==============================================================================
# TAB 1: VISUALIZACIONES
# ==============================================================================

with tab1:
    st.markdown("### Análisis Visual de Predicciones")
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
    <b>Storytelling:</b> Las visualizaciones te ayudan a identificar patrones, 
    anomalías y tendencias en las predicciones del modelo. Busca cambios bruscos, 
    sesgos inesperados o comportamientos anómalos.
    </div>
    """, unsafe_allow_html=True)
    
    # --- Fila 1: Distribución e Historial ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribución de Predicciones")
        st.markdown("""
        <small><i>
        ¿Las predicciones están balanceadas o sesgadas hacia un extremo?
        </i></small>
        """, unsafe_allow_html=True)
        
        # Histograma con Plotly
        fig_hist = px.histogram(
            logged_data,
            x='prediction',
            nbins=30,
            title="",
            labels={'prediction': 'Probabilidad Predicha', 'count': 'Frecuencia'},
            color_discrete_sequence=[COLOR_PALETTE['primary']]
        )
        
        # Agregar líneas de referencia
        fig_hist.add_vline(
            x=0.5, line_dash="dash", line_color="red",
            annotation_text="Umbral de decisión",
            annotation_position="top right"
        )
        
        fig_hist.add_vline(
            x=mean_prediction, line_dash="dot", line_color="green",
            annotation_text=f"Media: {mean_prediction:.3f}",
            annotation_position="top left"
        )
        
        fig_hist.update_layout(
            showlegend=False,
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Interpretación automática
        if mean_prediction > 0.7:
            st.success(
                "**Interpretación**: La mayoría de predicciones son altas "
                "(>0.7), indicando confianza en pagos a tiempo."
            )
        elif mean_prediction < 0.3:
            st.warning(
                "**Interpretación**: Muchas predicciones son bajas (<0.3), "
                "indicando preocupación por riesgo de impago."
            )
        else:
            st.info(
                "**Interpretación**: Distribución balanceada de predicciones, "
                "el modelo discrimina bien entre casos."
            )
    
    with col2:
        st.markdown("#### Evolución Temporal")
        st.markdown("""
        <small><i>
        ¿Hay tendencias o cambios en las predicciones a lo largo del tiempo?
        </i></small>
        """, unsafe_allow_html=True)
        
        if 'timestamp' in logged_data.columns:
            # Agrupar por intervalo de tiempo
            temporal_data = logged_data.set_index('timestamp').resample('5T')['prediction'].agg(['mean', 'std', 'count']).reset_index()
            
            # Gráfico de línea con banda de confianza
            fig_time = go.Figure()
            
            # Línea principal
            fig_time.add_trace(go.Scatter(
                x=temporal_data['timestamp'],
                y=temporal_data['mean'],
                mode='lines+markers',
                name='Predicción Media',
                line=dict(color=COLOR_PALETTE['secondary'], width=2),
                marker=dict(size=6)
            ))
            
            # Banda de confianza (±1 std)
            fig_time.add_trace(go.Scatter(
                x=temporal_data['timestamp'],
                y=temporal_data['mean'] + temporal_data['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_time.add_trace(go.Scatter(
                x=temporal_data['timestamp'],
                y=temporal_data['mean'] - temporal_data['std'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_time.update_layout(
                xaxis_title="Tiempo",
                yaxis_title="Predicción Media",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Detectar tendencias
            if len(temporal_data) > 2:
                first_mean = temporal_data['mean'].iloc[:3].mean()
                last_mean = temporal_data['mean'].iloc[-3:].mean()
                change = ((last_mean - first_mean) / first_mean) * 100
                
                if abs(change) > 10:
                    if change > 0:
                        st.info(
                            f"**Tendencia ascendente**: Las predicciones han "
                            f"aumentado {change:.1f}% → Menos casos de riesgo detectados"
                        )
                    else:
                        st.warning(
                            f"**Tendencia descendente**: Las predicciones han "
                            f"disminuido {abs(change):.1f}% → Más casos de riesgo detectados"
                        )
                else:
                    st.success("**Estabilidad**: No hay tendencias significativas")
        else:
            # Box plot alternativo
            fig_box = px.box(
                logged_data,
                y='prediction',
                title="",
                labels={'prediction': 'Probabilidad Predicha'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # --- Fila 2: Comparación con Referencia ---
    st.markdown("#### Comparación: Datos de Referencia vs Producción")
    st.markdown("""
    <small><i>
    ¿Los datos actuales son similares a los datos de entrenamiento? 
    Diferencias grandes pueden indicar drift.
    </i></small>
    """, unsafe_allow_html=True)
    
    # Seleccionar features numéricas para comparar
    numeric_cols = logged_data.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['prediction', 'timestamp']][:6]
    
    if len(numeric_cols) > 0:
        comparison_data = []
        
        for col in numeric_cols:
            if col in X_ref.columns:
                comparison_data.append({
                    'Feature': col,
                    'Referencia': X_ref[col].mean(),
                    'Producción': logged_data[col].mean() if col in logged_data.columns else 0,
                    'Diferencia_%': ((logged_data[col].mean() - X_ref[col].mean()) / X_ref[col].mean() * 100) if col in logged_data.columns and X_ref[col].mean() != 0 else 0
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Gráfico de barras agrupadas
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Bar(
                name='📚 Referencia (Entrenamiento)',
                x=comp_df['Feature'],
                y=comp_df['Referencia'],
                marker_color=COLOR_PALETTE['reference'],
                text=comp_df['Referencia'].round(2),
                textposition='outside'
            ))
            
            fig_comp.add_trace(go.Bar(
                name='🚀 Producción (Actual)',
                x=comp_df['Feature'],
                y=comp_df['Producción'],
                marker_color=COLOR_PALETTE['current'],
                text=comp_df['Producción'].round(2),
                textposition='outside'
            ))
            
            fig_comp.update_layout(
                title="Comparación de Medias: ¿Los datos han cambiado?",
                xaxis_title="Feature",
                yaxis_title="Valor Medio",
                barmode='group',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tabla de diferencias
            st.markdown("##### Tabla de Diferencias")
            
            # Aplicar formato condicional
            def highlight_diff(val):
                if abs(val) > 20:
                    return 'background-color: #ffcccb'  # Rojo claro
                elif abs(val) > 10:
                    return 'background-color: #ffffcc'  # Amarillo claro
                else:
                    return 'background-color: #ccffcc'  # Verde claro
            
            styled_df = comp_df.style.applymap(
                highlight_diff,
                subset=['Diferencia_%']
            ).format({
                'Referencia': '{:.2f}',
                'Producción': '{:.2f}',
                'Diferencia_%': '{:.1f}%'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Interpretación
            max_diff = comp_df['Diferencia_%'].abs().max()
            if max_diff > 20:
                st.error(
                    f"**Alerta de Drift**: Diferencias de hasta {max_diff:.1f}% detectadas. "
                    "Algunas features han cambiado significativamente."
                )
            elif max_diff > 10:
                st.warning(
                    f"**Drift Moderado**: Diferencias de hasta {max_diff:.1f}% detectadas. "
                    "Monitorear evolución."
                )
            else:
                st.success(
                    f"**Sin Drift Significativo**: Diferencias máximas de {max_diff:.1f}%. "
                    "Los datos se mantienen consistentes."
                )
    else:
        st.info("No hay features numéricas disponibles para comparación.")


# ==============================================================================
# TAB 2: ANÁLISIS DE DRIFT
# ==============================================================================

with tab2:
    st.markdown("### Análisis Detallado de Data Drift")
    
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
    <b>¿Qué es Data Drift?</b><br>
    Data drift ocurre cuando las características de los datos en producción 
    son significativamente diferentes a los datos de entrenamiento. 
    Esto puede degradar el desempeño del modelo porque "ve" datos fuera 
    de su experiencia de entrenamiento.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generando reporte de drift con Evidently AI..."):
        # Preparar datos para análisis
        current_data = logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
        
        # Asegurar que ambos datasets tengan las mismas columnas
        common_cols = list(set(X_ref.columns) & set(current_data.columns))
        
        if len(common_cols) == 0:
            st.error("No hay columnas en común entre datos de referencia y producción")
        else:
            drift_report = generate_drift_report(
                X_ref[common_cols],
                current_data[common_cols]
            )
            
            if drift_report:
                # Extraer métricas
                drift_metrics = extract_drift_metrics(drift_report)
                
                # --- Mostrar métricas resumen ---
                st.markdown("#### Resumen de Drift")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    drift_detected = drift_metrics.get('dataset_drift', False)
                    st.metric(
                        "Drift en Dataset",
                        "SÍ" if drift_detected else "NO",
                        delta="Crítico" if drift_detected else "Normal",
                        delta_color="inverse" if drift_detected else "off"
                    )
                
                with col2:
                    drifted = drift_metrics.get('drifted_features', 0)
                    st.metric(
                        "Features con Drift",
                        f"{drifted}",
                        help="Número de features que presentan drift estadísticamente significativo"
                    )
                
                with col3:
                    total = drift_metrics.get('total_features', 0)
                    st.metric(
                        "Total Features",
                        f"{total}",
                        help="Total de features analizadas"
                    )
                
                with col4:
                    pct = drift_metrics.get('drift_percentage', 0)
                    st.metric(
                        "% con Drift",
                        f"{pct:.1f}%",
                        help="Porcentaje de features que presentan drift"
                    )
                
                # --- Interpretación ---
                st.markdown("---")
                st.markdown("#### Interpretación y Acciones Recomendadas")
                
                if drift_metrics.get('dataset_drift'):
                    st.error("""
                    **ALERTA: Drift Significativo Detectado**
                    
                    El análisis estadístico indica que los datos actuales son 
                    significativamente diferentes a los datos de entrenamiento.
                    
                    **Impactos Potenciales:**
                    - Degradación del desempeño del modelo
                    - Predicciones menos confiables
                    - Aumento de errores en producción
                    
                    **Acciones Inmediatas:**
                    1. **Re-entrenar el modelo** con datos recientes
                    2. **Investigar causas** del drift (¿cambios en el negocio?)
                    3. **Validar calidad** de los datos actuales
                    4. **Consultar con stakeholders** sobre cambios conocidos
                    """)
                else:
                    if pct > 30:
                        st.warning("""
                        **Drift Moderado Detectado**
                        
                        Aunque no hay drift a nivel de dataset, varias features 
                        individuales presentan cambios.
                        
                        **Recomendaciones:**
                        - Monitorear de cerca la evolución
                        - Revisar features específicas con drift
                        - Planificar re-entrenamiento preventivo
                        """)
                    else:
                        st.success("""
                        **Sin Drift Significativo**
                        
                        Los datos en producción son consistentes con los datos 
                        de entrenamiento. El modelo opera en su rango esperado.
                        
                        **Mantenimiento:**
                        - Continuar monitoreo regular
                        - Revisar métricas periódicamente
                        - Re-entrenar según calendario establecido
                        """)
                
                # --- Reporte visual de Evidently ---
                st.markdown("---")
                st.markdown("#### Reporte Detallado de Evidently AI")
                
                try:
                    # Intentar mostrar reporte HTML
                    st.components.v1.html(
                        drift_report._repr_html_(),
                        height=1000,
                        scrolling=True
                    )
                except Exception as e:
                    st.warning(f"No se pudo renderizar el reporte HTML: {str(e)}")
                    st.info("El reporte se generó correctamente pero no puede mostrarse en la interfaz")
            else:
                st.error("No se pudo generar el reporte de drift")


# ==============================================================================
# TAB 3: LOG DE DATOS
# ==============================================================================

with tab3:
    st.markdown("### Log Histórico de Predicciones")
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
    <b>Registro Completo:</b> Este log contiene todas las predicciones 
    realizadas por el modelo en producción, incluyendo features de entrada, 
    predicciones y timestamps. Útil para auditoría y análisis retrospectivo.
    </div>
    """, unsafe_allow_html=True)
    
    # --- Controles de visualización ---
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        show_rows = st.selectbox(
            "Mostrar últimas:",
            [10, 25, 50, 100, 200, 500],
            index=1,
            help="Número de filas a mostrar en la tabla"
        )
    
    with col2:
        sort_order = st.radio(
            "Orden:",
            ["Más recientes primero", "Más antiguas primero"],
            horizontal=True
        )
    
    # --- Estadísticas del log ---
    st.markdown("#### Estadísticas del Log")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Registros", f"{len(logged_data):,}")
    
    with col2:
        if 'timestamp' in logged_data.columns:
            first_date = logged_data['timestamp'].min()
            st.metric("Primer Registro", first_date.strftime("%Y-%m-%d %H:%M"))
    
    with col3:
        if 'timestamp' in logged_data.columns:
            last_date = logged_data['timestamp'].max()
            st.metric("Último Registro", last_date.strftime("%Y-%m-%d %H:%M"))
    
    with col4:
        if 'timestamp' in logged_data.columns:
            duration = (logged_data['timestamp'].max() - logged_data['timestamp'].min())
            hours = duration.total_seconds() / 3600
            st.metric("Duración", f"{hours:.1f}h")
    
    st.markdown("---")
    
    # --- Tabla de datos ---
    st.markdown("#### Datos Detallados")
    
    # Ordenar datos
    if sort_order == "Más recientes primero":
        display_data = logged_data.tail(show_rows).iloc[::-1]
    else:
        display_data = logged_data.head(show_rows)
    
    # Mostrar tabla con formato
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # --- Botón de descarga ---
    st.markdown("---")
    st.markdown("#### Descargar Datos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Descarga el log completo en formato CSV para análisis offline, 
        reportes o respaldo.
        """)
    
    with col2:
        csv = logged_data.to_csv(index=False)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"monitoring_log_{timestamp}.csv",
            mime="text/csv",
            type="primary"
        )


# ==============================================================================
# TAB 4: INSIGHTS ACCIONABLES
# ==============================================================================

with tab4:
    st.markdown("### Insights Accionables y Recomendaciones")
    
    st.markdown("""
    <div style='background-color: #d1ecf1; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
    <b>Objetivo:</b> Este panel sintetiza los hallazgos más importantes 
    y proporciona recomendaciones concretas para acción inmediata.
    </div>
    """, unsafe_allow_html=True)
    
    # --- Análisis Automático ---
    
    insights = []
    recommendations = []
    
    # 1. Análisis de volumen
    if total_predictions < 100:
        insights.append(("Bajo Volumen de Datos", 
                        f"Solo {total_predictions} predicciones registradas. "
                        "Se necesitan más datos para análisis robusto."))
        recommendations.append(("Recolectar Más Datos", 
                               "Genera al menos 500-1000 predicciones para análisis estadísticamente significativo."))
    
    # 2. Análisis de distribución
    if std_prediction < 0.1:
        insights.append(("Baja Discriminación", 
                        f"Desviación estándar muy baja ({std_prediction:.3f}). "
                        "El modelo puede no estar diferenciando bien entre casos."))
        recommendations.append(("Revisar Modelo", 
                               "Considera re-entrenar o ajustar hiperparámetros para mejorar discriminación."))
    
    if risky_rate > 20:
        insights.append(("Alta Tasa de Riesgo", 
                        f"{risky_rate:.1f}% de clientes clasificados como riesgosos. "
                        "Esto es significativamente alto."))
        recommendations.append(("Investigar Causas", 
                               "- ¿Ha cambiado el perfil de clientes?\n"
                               "- ¿Hay factores externos (economía, regulaciones)?\n"
                               "- ¿El modelo necesita calibración?"))
    
    # 3. Análisis de drift (si está disponible)
    if 'drift_metrics' in locals():
        if drift_metrics.get('dataset_drift'):
            insights.append(("Drift Crítico Detectado", 
                            f"{drift_metrics.get('drift_percentage', 0):.1f}% de features con drift. "
                            "Los datos han cambiado significativamente."))
            recommendations.append(("Re-entrenamiento Urgente", 
                                   "1. Recolectar datos recientes\n"
                                   "2. Re-entrenar modelo\n"
                                   "3. Validar desempeño antes de desplegar\n"
                                   "4. Implementar monitoreo continuo"))
    
    # 4. Análisis temporal
    if 'timestamp' in logged_data.columns and len(logged_data) > 10:
        recent = logged_data.tail(100)['prediction'].mean()
        old = logged_data.head(100)['prediction'].mean()
        change = ((recent - old) / old) * 100
        
        if abs(change) > 15:
            direction = "aumentado" if change > 0 else "disminuido"
            insights.append((f"Tendencia {direction.capitalize()}", 
                            f"Las predicciones han {direction} {abs(change):.1f}% en el tiempo."))
            recommendations.append(("Monitorear Tendencia", 
                                   f"Investiga por qué las predicciones están {direction}. "
                                   "Puede indicar cambios en el negocio o en los datos."))
    
    # 5. Salud general del sistema
    if not insights:
        insights.append(("Sistema Operando Normalmente", 
                        "No se detectaron problemas críticos. "
                        "El modelo está operando dentro de parámetros esperados."))
        recommendations.append(("Mantenimiento Preventivo", 
                               "- Continuar monitoreo regular\n"
                               "- Planificar re-entrenamiento trimestral\n"
                               "- Documentar cambios en el negocio\n"
                               "- Mantener log de predicciones actualizado"))
    
    # --- Mostrar Insights ---
    st.markdown("#### Hallazgos Principales")
    
    for i, (title, description) in enumerate(insights, 1):
        with st.expander(f"{i}. {title}", expanded=True):
            st.markdown(description)
    
    st.markdown("---")
    
    # --- Mostrar Recomendaciones ---
    st.markdown("#### Recomendaciones Accionables")
    
    for i, (title, description) in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style='background-color: #fff3cd; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
        <h4 style='margin-top: 0;'>{i}. {title}</h4>
        <p style='margin-bottom: 0; white-space: pre-line;'>{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Checklist de Acción ---
    st.markdown("#### Checklist de Acción")
    
    st.markdown("""
    Utiliza este checklist para asegurar que estás tomando las acciones necesarias:
    """)
    
    checklist_items = [
        "Revisar KPIs principales semanalmente",
        "Generar predicciones regularmente (mínimo diario)",
        "Analizar reporte de drift mensualmente",
        "Documentar anomalías o cambios observados",
        "Comunicar hallazgos a stakeholders",
        "Planificar re-entrenamiento si hay drift >50%",
        "Mantener backup del log de predicciones",
        "Validar que la API esté funcionando correctamente",
        "Comparar predicciones con resultados reales (cuando disponibles)",
        "Actualizar documentación del modelo"
    ]
    
    for item in checklist_items:
        st.checkbox(item, key=f"check_{item}")


# ==============================================================================
# 14. FOOTER CON INFORMACIÓN DEL SISTEMA
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Dashboard de Monitoreo ML v2.0</b></p>
    <p>Sistema de Analytics | Última actualización: {timestamp}</p>
    <p>
        <small>
        <b>Tip:</b> Para mejores resultados, genera predicciones regularmente 
        y revisa este dashboard al menos semanalmente.
        </small>
    </p>
</div>
""".format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), 
unsafe_allow_html=True)


# ==============================================================================
# FIN DEL DASHBOARD
# ==============================================================================
