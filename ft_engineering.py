"""
==============================================================================
MÓDULO DE FEATURE ENGINEERING Y PREPROCESAMIENTO DE DATOS
==============================================================================

CONTEXTO DEL NEGOCIO:
--------------------
El preprocesamiento de datos es el puente entre datos crudos y modelos efectivos.
En nuestro caso de predicción de pagos a tiempo, necesitamos transformar datos
heterogéneos (numéricos, categóricos, con valores faltantes) en un formato que
los algoritmos de ML puedan procesar eficientemente.

¿POR QUÉ ES CRÍTICO EL PREPROCESAMIENTO?
---------------------------------------
1. **Calidad de Datos**: "Garbage in, garbage out"
   - Valores faltantes pueden causar errores o sesgos
   - Escalas diferentes afectan algoritmos basados en distancia
   
2. **Compatibilidad**: Los modelos ML requieren entradas numéricas
   - Variables categóricas deben convertirse (encoding)
   - Formato consistente entre entrenamiento y producción
   
3. **Desempeño del Modelo**: El preprocesamiento impacta directamente
   - Imputación inteligente preserva información
   - Encoding adecuado captura relaciones categóricas
   - Pipelines aseguran reproducibilidad

ARQUITECTURA DEL PIPELINE:
-------------------------
Este módulo implementa un pipeline de preprocesamiento con dos rutas paralelas:

┌─────────────────────────────────────────────┐
│           DATOS CRUDOS (Mixed)              │
└───────────────┬─────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
    ┌───▼────┐      ┌───▼────┐
    │ Numeric│      │Categoric│
    │Features│      │Features │
    └───┬────┘      └───┬────┘
        │               │
    ┌───▼────┐      ┌───▼────────┐
    │Imputer │      │  to_str    │
    │ (mean) │      │  Imputer   │
    └───┬────┘      │ OneHot Enc │
        │           └───┬────────┘
        │               │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ UNIFIED ARRAY │
        │   (Numeric)   │
        └───────────────┘

DECISIONES DE DISEÑO:
--------------------
1. **Imputación por Media (Numéricas)**: 
   - Preserva la distribución general
   - No introduce sesgos extremos
   - Alternativas: mediana (más robusta), KNN (más sofisticada)

2. **Imputación por Moda (Categóricas)**:
   - Valor más frecuente es "mejor guess"
   - Minimiza error en clasificación
   
3. **One-Hot Encoding (Categóricas)**:
   - No asume orden en categorías
   - Evita sesgos de encoding ordinal incorrecto
   - Trade-off: aumenta dimensionalidad

MEJORAS FUTURAS POTENCIALES:
---------------------------
- Scaling/Normalización para algoritmos sensibles a escala
- Feature selection para reducir dimensionalidad
- Feature engineering (ratios, interacciones, transformaciones)
- Análisis de importancia de features
- Detección de outliers y tratamiento especializado

Autor: Santiago Joaquin Mozo
Versión: 2.0
Fecha: 2026
==============================================================================
"""

# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================

# --- Análisis de datos ---
import pandas as pd
import numpy as np

# --- Preprocesamiento de sklearn ---
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# --- Utilidades ---
from typing import Optional, Tuple, List
import warnings

# Suprimir warnings de futuras versiones (opcional)
warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

# Ruta al archivo de datos
DATA_PATH = "base_de_datoslimpia.csv"

# Configuración de reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Nombre de la columna target
TARGET_COLUMN = "Pago_atiempo"

# Estrategias de imputación
NUMERIC_IMPUTATION_STRATEGY = "mean"    # Alternativas: 'median', 'constant'
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"  # Alternativa: 'constant'

# Configuración de encoding
HANDLE_UNKNOWN_CATEGORIES = "ignore"  # Ignora categorías no vistas en train
SPARSE_OUTPUT = False  # False para compatibilidad con la mayoría de modelos


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXPLORACIÓN
# ==============================================================================

def load_data(show_info: bool = False) -> pd.DataFrame:
    """
    Carga el dataset limpio desde CSV.
    
    PROPÓSITO:
    ---------
    Esta función centraliza la carga de datos y opcionalmente muestra
    información exploratoria para entender la estructura antes del
    preprocesamiento.
    
    Args:
        show_info: Si True, muestra información exploratoria del dataset
        
    Returns:
        pd.DataFrame: Dataset cargado
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo
        ValueError: Si el archivo está vacío o corrupto
    """
    try:
        # Cargar CSV
        df = pd.read_csv(DATA_PATH)
        
        # Validar que no esté vacío
        if df.empty:
            raise ValueError(f"El archivo '{DATA_PATH}' está vacío")
        
        # Validar que exista la columna target
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"La columna target '{TARGET_COLUMN}' no existe en el dataset. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        # Mostrar información exploratoria si se solicita
        if show_info:
            print("\n" + "="*80)
            print("INFORMACIÓN DEL DATASET")
            print("="*80 + "\n")
            
            print(f"Archivo cargado: {DATA_PATH}")
            print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")
            
            print("ESTRUCTURA DEL DATASET:")
            print("-" * 80)
            df.info()
            
            print("\nPRIMERAS FILAS:")
            print("-" * 80)
            print(df.head())
            
            print("\nESTADÍSTICAS DESCRIPTIVAS:")
            print("-" * 80)
            print(df.describe())
            
            # Análisis de valores faltantes
            missing = df.isnull().sum()
            if missing.any():
                print("\nVALORES FALTANTES DETECTADOS:")
                print("-" * 80)
                missing_df = pd.DataFrame({
                    'Columna': missing[missing > 0].index,
                    'Faltantes': missing[missing > 0].values,
                    'Porcentaje': (missing[missing > 0] / len(df) * 100).round(2)
                })
                print(missing_df.to_string(index=False))
                print(f"\nEstos valores serán imputados durante el preprocesamiento")
            else:
                print("\nNo hay valores faltantes en el dataset")
            
            print("\n" + "="*80 + "\n")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: No se encontró el archivo '{DATA_PATH}'\n"
            f"   Verifica que el archivo exista en el directorio del proyecto."
        )
    except Exception as e:
        raise ValueError(f"Error al cargar datos: {str(e)}")


def analyze_feature_types(X: pd.DataFrame, verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    Identifica y categoriza features numéricas y categóricas.
    
    IMPORTANCIA DE LA SEPARACIÓN:
    -----------------------------
    Diferentes tipos de datos requieren diferentes transformaciones:
    
    - **Numéricas**: Imputación por estadísticos (mean, median)
    - **Categóricas**: Imputación por moda + Encoding
    
    La detección automática permite que el pipeline sea robusto ante
    cambios en el esquema de datos.
    
    Args:
        X: DataFrame con features
        verbose: Si True, muestra resumen de la clasificación
        
    Returns:
        Tuple[List[str], List[str]]: (columnas_numéricas, columnas_categóricas)
    """
    # Identificar tipos
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    if verbose:
        print("\n" + "="*80)
        print("ANÁLISIS DE TIPOS DE FEATURES")
        print("="*80 + "\n")
        
        print(f"Features Numéricas ({len(numeric_features)}):")
        print("-" * 80)
        if numeric_features:
            for i, feat in enumerate(numeric_features, 1):
                print(f"   {i:2}. {feat}")
        else:
            print("   (Ninguna)")
        
        print(f"\nFeatures Categóricas ({len(categorical_features)}):")
        print("-" * 80)
        if categorical_features:
            for i, feat in enumerate(categorical_features, 1):
                print(f"   {i:2}. {feat}")
        else:
            print("   (Ninguna)")
        
        print("\nESTRATEGIA DE PREPROCESAMIENTO:")
        print("-" * 80)
        print(f"   - Numericas: Imputacion por {NUMERIC_IMPUTATION_STRATEGY}")
        print(f"   - Categoricas: Imputacion por {CATEGORICAL_IMPUTATION_STRATEGY} + One-Hot Encoding")
        print("\n" + "="*80 + "\n")
    
    return numeric_features, categorical_features


# ==============================================================================
# 4. CONSTRUCCIÓN DE PIPELINES DE TRANSFORMACIÓN
# ==============================================================================

def create_numeric_transformer() -> Pipeline:
    """
    Crea el pipeline de transformación para features numéricas.
    
    PIPELINE NUMÉRICO:
    -----------------
    1. **SimpleImputer (mean)**: Rellena valores faltantes con la media
       
       ¿Por qué la media?
       - Preserva la suma total de los datos
       - No introduce sesgos hacia extremos
       - Funciona bien con distribuciones normales
       
       Consideraciones:
       - Sensible a outliers (alternativa: mediana)
       - Puede no ser óptima para distribuciones sesgadas
    
    POSIBLES EXTENSIONES:
    --------------------
    - StandardScaler: Para normalizar escala (Z-score)
    - MinMaxScaler: Para escalar a rango [0,1]
    - RobustScaler: Más robusto a outliers
    - PolynomialFeatures: Para capturar interacciones
    
    Returns:
        Pipeline: Pipeline configurado para transformación numérica
    """
    return Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(
                    strategy=NUMERIC_IMPUTATION_STRATEGY,
                    add_indicator=False  # No agregar columnas de indicador de faltante
                )
            ),
            # Aquí podrían agregarse más pasos:
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.95)),
        ],
        verbose=False
    )


def create_categorical_transformer() -> Pipeline:
    """
    Crea el pipeline de transformación para features categóricas.
    
    PIPELINE CATEGÓRICO:
    -------------------
    1. **FunctionTransformer (to_str)**: Asegura que todo sea string
       - Previene errores de tipo de dato
       - Maneja casos donde categorías son números
       
    2. **SimpleImputer (most_frequent)**: Rellena con la moda
       - La categoría más común es la "mejor apuesta"
       - Minimiza error de clasificación esperado
       - No introduce categorías artificiales
       
    3. **OneHotEncoder**: Convierte categorías en columnas binarias
       
       ¿Por qué One-Hot?
       - No asume orden entre categorías
       - Cada categoría es tratada independientemente
       - Compatible con la mayoría de modelos ML
       
       Ejemplo:
       Color: ['Rojo', 'Azul', 'Verde']
       →
       Color_Rojo: [1, 0, 0]
       Color_Azul: [0, 1, 0]
       Color_Verde: [0, 0, 1]
       
       Trade-offs:
       - ✅ No introduce sesgos ordinales
       - ✅ Flexible y general
       - ❌ Aumenta dimensionalidad (problema con muchas categorías)
       - ❌ Curse of dimensionality con variables de alta cardinalidad
       
       Alternativas:
       - Label Encoding: Para variables ordinales
       - Target Encoding: Usa información del target (cuidado con leakage)
       - Embedding Layers: Para deep learning
    
    Returns:
        Pipeline: Pipeline configurado para transformación categórica
    """
    return Pipeline(
        steps=[
            (
                'to_string',
                FunctionTransformer(
                    func=lambda x: x.astype(str),
                    validate=False
                )
            ),
            (
                'imputer',
                SimpleImputer(
                    strategy=CATEGORICAL_IMPUTATION_STRATEGY,
                    add_indicator=False
                )
            ),
            (
                'onehot',
                OneHotEncoder(
                    handle_unknown=HANDLE_UNKNOWN_CATEGORIES,  # Ignora categorías nuevas
                    sparse_output=SPARSE_OUTPUT,  # Matriz densa para compatibilidad
                    drop=None  # Mantiene todas las categorías (alternativa: 'first' para evitar multicolinealidad)
                )
            ),
        ],
        verbose=False
    )


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Crea el ColumnTransformer que orquesta todas las transformaciones.
    
    ¿QUÉ ES UN COLUMNTRANSFORMER?
    -----------------------------
    Es un orquestador que aplica diferentes transformaciones a diferentes
    conjuntos de columnas en paralelo, luego concatena los resultados.
    
    Ventajas:
    - ✅ Evita data leakage (fit solo en train, transform en train/test)
    - ✅ Reproducibilidad garantizada
    - ✅ Facilita deployment (un solo objeto serializable)
    - ✅ Código limpio y mantenible
    
    FLUJO DE PROCESAMIENTO:
    ----------------------
    DataFrame Original
         ↓
    ColumnTransformer
         ↓
    ┌─────────┬──────────────┐
    │ Numeric │ Categorical  │
    │ Pipeline│ Pipeline     │
    └─────────┴──────────────┘
         ↓
    Concatenación Horizontal
         ↓
    Matriz Numérica Unificada
    
    Args:
        numeric_features: Lista de nombres de columnas numéricas
        categorical_features: Lista de nombres de columnas categóricas
        
    Returns:
        ColumnTransformer: Transformador completo configurado
    """
    # Crear transformadores individuales
    numeric_transformer = create_numeric_transformer()
    categorical_transformer = create_categorical_transformer()
    
    # Lista de transformadores a aplicar
    transformers = []
    
    # Agregar transformador numérico si hay features numéricas
    if numeric_features:
        transformers.append(
            ('numeric', numeric_transformer, numeric_features)
        )
    
    # Agregar transformador categórico si hay features categóricas
    if categorical_features:
        transformers.append(
            ('categorical', categorical_transformer, categorical_features)
        )
    
    # Validar que hay al menos un transformador
    if not transformers:
        raise ValueError(
            "Error: No se detectaron features numéricas ni categóricas. "
            "Verifica el dataset."
        )
    
    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Descarta columnas no especificadas
        sparse_threshold=0.0,  # No usar matrices sparse
        n_jobs=1,  # Paralelización (1=sin paralelizar, -1=todos los cores)
        verbose=False
    )
    
    return preprocessor


# ==============================================================================
# 5. FUNCIÓN PRINCIPAL: ft_engineering()
# ==============================================================================

def ft_engineering(X_sample: Optional[pd.DataFrame] = None) -> ColumnTransformer:
    """
    Función principal de feature engineering: crea un preprocessor configurado.
    
    PROPÓSITO:
    ---------
    Esta es la función "exportable" del módulo. Otros scripts la importan
    para obtener un preprocessor consistente configurado según las
    características del dataset.
    
    FLUJO DE TRABAJO TÍPICO:
    ------------------------
    1. Entrenamiento:
       ```python
       preprocessor = ft_engineering(X_train)
       X_train_transformed = preprocessor.fit_transform(X_train)
       X_test_transformed = preprocessor.transform(X_test)
       model.fit(X_train_transformed, y_train)
       ```
    
    2. Producción:
       ```python
       preprocessor = ft_engineering(X_ref)
       preprocessor.fit(X_ref)  # Ajustar a datos históricos
       X_new_transformed = preprocessor.transform(X_new)
       predictions = model.predict(X_new_transformed)
       ```
    
    GARANTÍAS:
    ----------
    - ✅ El preprocessor es stateless (puede aplicarse múltiples veces)
    - ✅ Fit debe ejecutarse SOLO en datos de entrenamiento
    - ✅ Transform puede aplicarse en train, test, producción
    - ✅ Las transformaciones son reproducibles
    
    Args:
        X_sample: DataFrame de muestra para determinar tipos de features.
                  Si es None, carga el dataset completo.
        
    Returns:
        ColumnTransformer: Preprocessor configurado y listo para fit/transform
        
    Raises:
        ValueError: Si no se pueden determinar los tipos de features
    
    Examples:
        >>> from ft_engineering import ft_engineering
        >>> 
        >>> # Opción 1: Con muestra
        >>> preprocessor = ft_engineering(X_train)
        >>> X_processed = preprocessor.fit_transform(X_train)
        >>> 
        >>> # Opción 2: Sin muestra (usa dataset completo)
        >>> preprocessor = ft_engineering()
        >>> X_processed = preprocessor.fit_transform(X_train)
    """
    # Si no se proporciona muestra, cargar dataset completo
    if X_sample is None:
        df = load_data(show_info=False)
        X_sample = df.drop(columns=[TARGET_COLUMN])
    
    # Identificar tipos de features
    numeric_features, categorical_features = analyze_feature_types(
        X_sample,
        verbose=False
    )
    
    # Crear y retornar preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    return preprocessor


# ==============================================================================
# 6. DEMOSTRACIÓN Y VALIDACIÓN (Script Principal)
# ==============================================================================

if __name__ == "__main__":
    """
    Bloque de demostración que se ejecuta cuando el script se corre directamente.
    
    PROPÓSITO EDUCATIVO:
    -------------------
    Este bloque demuestra cómo usar el módulo y valida que todo funcione
    correctamente. También sirve como documentación ejecutable.
    
    ¿Cuándo se ejecuta?
    - Cuando corres: python ft_engineering.py
    
    ¿Cuándo NO se ejecuta?
    - Cuando importas: from ft_engineering import ft_engineering
    """
    
    print("\n" + "="*80)
    print("DEMOSTRACIÓN DE FEATURE ENGINEERING")
    print("="*80)
    
    # --- PASO 1: Cargar datos ---
    print("\nPASO 1: Cargando datos...")
    print("-" * 80)
    
    df = load_data(show_info=True)
    
    # --- PASO 2: Separar features y target ---
    print("\nPASO 2: Separando features y target...")
    print("-" * 80)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    print(f"   Features (X): {X.shape[1]} columnas")
    print(f"   Target (y): {y.name}")
    print(f"   Distribución del target:")
    print(f"      - Clase 0 (No pago a tiempo): {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
    print(f"      - Clase 1 (Pago a tiempo): {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    
    # --- PASO 3: Analizar tipos de features ---
    print("\nPASO 3: Analizando tipos de features...")
    print("-" * 80)
    
    numeric_features, categorical_features = analyze_feature_types(X, verbose=True)
    
    # --- PASO 4: Dividir datos ---
    print("\nPASO 4: Dividiendo datos en train/test...")
    print("-" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]:,} muestras ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   Test:  {X_test.shape[0]:,} muestras ({TEST_SIZE*100:.0f}%)")
    print(f"   División estratificada aplicada")
    
    # --- PASO 5: Crear preprocessor ---
    print("\nPASO 5: Creando preprocessor...")
    print("-" * 80)
    
    preprocessor = ft_engineering(X_train)
    
    print(f"   Preprocessor creado exitosamente")
    print(f"   Transformadores configurados:")
    print(f"      - Numéricos: {len(numeric_features)} features")
    print(f"      - Categóricos: {len(categorical_features)} features")
    
    # --- PASO 6: Aplicar transformaciones ---
    print("\nPASO 6: Aplicando transformaciones...")
    print("-" * 80)
    
    print("   - Ajustando preprocessor en datos de entrenamiento (fit)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("   - Transformando datos de prueba (transform)...")
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\n   Transformaciones completadas")
    
    # --- PASO 7: Resultados ---
    print("\nPASO 7: Resultados del preprocesamiento")
    print("="*80)
    
    print("\nDATOS DE ENTRENAMIENTO:")
    print("-" * 80)
    print(f"   - Shape original:      {X_train.shape}")
    print(f"   - Shape procesado:     {X_train_processed.shape}")
    print(f"   - Tipo de datos:       {type(X_train_processed)}")
    print(f"   - Dtype:               {X_train_processed.dtype if hasattr(X_train_processed, 'dtype') else 'numpy.ndarray'}")
    
    print("\nDATOS DE PRUEBA:")
    print("-" * 80)
    print(f"   - Shape original:      {X_test.shape}")
    print(f"   - Shape procesado:     {X_test_processed.shape}")
    print(f"   - Tipo de datos:       {type(X_test_processed)}")
    print(f"   - Dtype:               {X_test_processed.dtype if hasattr(X_test_processed, 'dtype') else 'numpy.ndarray'}")
    
    # Análisis de dimensionalidad
    original_features = X_train.shape[1]
    processed_features = X_train_processed.shape[1]
    feature_increase = processed_features - original_features
    
    print("\nANÁLISIS DE DIMENSIONALIDAD:")
    print("-" * 80)
    print(f"   - Features originales:     {original_features}")
    print(f"   - Features procesadas:     {processed_features}")
    print(f"   - Incremento:              +{feature_increase} ({(feature_increase/original_features)*100:.1f}%)")
    
    if feature_increase > 0:
        print(f"\n   El incremento se debe al One-Hot Encoding de variables categóricas.")
        print(f"      Cada categoría única se convierte en una columna binaria.")
    
    # Muestra de datos procesados
    print("\nMUESTRA DE DATOS PROCESADOS (primeras 5 filas, primeras 10 columnas):")
    print("-" * 80)
    
    sample_data = X_train_processed[:5, :min(10, X_train_processed.shape[1])]
    print(sample_data)
    
    # Verificar valores faltantes
    if hasattr(X_train_processed, 'isnull'):
        missing_after = X_train_processed.isnull().sum().sum()
    else:
        missing_after = np.isnan(X_train_processed).sum()
    
    print(f"\nVALIDACIÓN:")
    print("-" * 80)
    print(f"   - Valores faltantes después del preprocesamiento: {missing_after}")
    
    if missing_after == 0:
        print(f"   ¡Perfecto! Todos los valores faltantes fueron imputados correctamente")
    else:
        print(f"   Advertencia: Aún hay valores faltantes. Revisa la configuración.")
    
    # --- PASO 8: Interpretación y conclusiones ---
    print("\n" + "="*80)
    print("INTERPRETACIÓN Y CONCLUSIONES")
    print("="*80 + "\n")
    
    print("STORYTELLING CON LOS DATOS:")
    print("-" * 80)
    print("""
El pipeline de preprocesamiento transforma datos heterogeneos y potencialmente
incompletos en una matriz numerica lista para modelos de Machine Learning.

QUE LOGRAMOS?
-----------------
1. Manejo robusto de valores faltantes
   - Imputacion inteligente preserva informacion sin introducir sesgos

2. Conversion de variables categoricas
   - One-Hot Encoding permite que el modelo capture patrones categoricos

3. Formato unificado
   - Todos los datos en matriz numerica, lista para cualquier algoritmo ML

4. Reproducibilidad garantizada
   - El mismo pipeline puede aplicarse en produccion

IMPACTO EN EL MODELO:
-----------------------
- Sin preprocesamiento: El modelo no puede entrenarse o tiene errores
- Con preprocesamiento: El modelo puede aprender patrones complejos

FLUJO EN PRODUCCION:
----------------------
1. Datos nuevos llegan -> 2. Se aplica preprocessor.transform() ->
3. Prediccion del modelo -> 4. Decision de negocio

CONSIDERACIONES IMPORTANTES:
--------------------------------
- El preprocessor debe ajustarse (fit) SOLO en datos de entrenamiento
- En produccion, usar el MISMO preprocessor ajustado en train
- Nunca hacer fit en datos de test o produccion (causa data leakage)

PROXIMOS PASOS:
-----------------
1. Usar este preprocessor en model_training_evaluation.py
2. Serializar el preprocessor junto con el modelo (joblib/pickle)
3. Monitorear si la distribucion de features cambia en produccion
4. Considerar feature engineering adicional segun performance del modelo
    """)
    
    print("\n" + "="*80)
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80)
    print(f"""
EXPORTABLE: La función ft_engineering() está lista para uso en otros módulos.

Ejemplo de uso:
    >>> from ft_engineering import ft_engineering
    >>> preprocessor = ft_engineering(X_train)
    >>> X_transformed = preprocessor.fit_transform(X_train)
    
""")
