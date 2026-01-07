"""
==============================================================================
SISTEMA DE ENTRENAMIENTO Y EVALUACIÓN DE MODELOS PREDICTIVOS
==============================================================================

CONTEXTO DEL NEGOCIO:
--------------------
Este módulo implementa un sistema completo para predecir si un cliente
pagará a tiempo sus obligaciones financieras. La capacidad de identificar
clientes con riesgo de pago tardío permite:

- Reducir pérdidas por morosidad
- Optimizar políticas de crédito
- Mejorar la gestión de riesgos
- Tomar decisiones proactivas sobre cobranza

ARQUITECTURA DEL SISTEMA:
------------------------
1. Preparación de datos y feature engineering
2. Entrenamiento de múltiples modelos (ensemble approach)
3. Evaluación exhaustiva con validación cruzada
4. Visualización interpretable de resultados
5. Selección del mejor modelo basada en métricas de negocio

MÉTRICA CLAVE DE NEGOCIO:
------------------------
Priorizamos el RECALL para la clase de "No pago a tiempo" porque:
- Es más costoso NO identificar un cliente riesgoso (falso negativo)
- Que clasificar erróneamente a un buen cliente (falso positivo)
- Un alto recall asegura que capturemos la mayoría de casos problemáticos

Autor: Sistema de Analytics
Fecha: 2026
==============================================================================
"""

# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y DEPENDENCIAS
# ==============================================================================

# --- Librerías core de Machine Learning ---
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --- Modelos de clasificación ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb

# --- Visualización y análisis ---
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Módulos personalizados del proyecto ---
from ft_engineering import ft_engineering


# ==============================================================================
# 2. CONFIGURACIÓN Y CONSTANTES DEL PROYECTO
# ==============================================================================

# Ruta al archivo de datos
DATA_PATH = "base_de_datoslimpia.csv"

# Configuración de reproducibilidad para resultados consistentes
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuración de visualización para gráficos más profesionales
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Parámetros del proyecto
TEST_SIZE = 0.2
N_CROSS_VALIDATION_FOLDS = 10
N_LEARNING_CURVE_SPLITS = 50

# Métricas que monitoreamos (alineadas con objetivos de negocio)
SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Etiqueta para la clase positiva (0 = No pago a tiempo)
POSITIVE_LABEL = 0


# ==============================================================================
# 3. CARGA Y PREPARACIÓN INICIAL DE DATOS
# ==============================================================================

print("\n" + "="*80)
print("INICIANDO PIPELINE DE MACHINE LEARNING")
print("="*80 + "\n")

print("Paso 1: Cargando datos del sistema...")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"   Archivo cargado: {DATA_PATH}")
except FileNotFoundError:
    print(f"   Error: No se encontró el archivo '{DATA_PATH}'")
    print(f"      Verifica que el archivo exista en el directorio actual")
    raise
except Exception as e:
    print(f"   Error al cargar datos: {str(e)}")
    raise

print(f"   Dataset cargado exitosamente: {df.shape[0]} registros, {df.shape[1]} columnas")
print(f"   Distribución de clases:")

# Análisis de balance de clases (crítico para entender el problema)
class_distribution = df['Pago_atiempo'].value_counts()
print(f"      - Pago a tiempo (1): {class_distribution.get(1, 0)} ({class_distribution.get(1, 0)/len(df)*100:.2f}%)")
print(f"      - No pago a tiempo (0): {class_distribution.get(0, 0)} ({class_distribution.get(0, 0)/len(df)*100:.2f}%)")

# INSIGHT: El desbalance de clases requiere estrategias especiales
if class_distribution.get(0, 0) / len(df) < 0.1:
    print("\n   ALERTA: Dataset altamente desbalanceado detectado")
    print("      - Se aplicaran tecnicas de balanceo (class_weight='balanced')")
    print("      - Metricas como accuracy pueden ser enganosas")
    print("      - Priorizar precision, recall y F1-score para clase minoritaria\n")


# ==============================================================================
# 4. SEPARACIÓN DE FEATURES Y TARGET
# ==============================================================================

print("\nPaso 2: Preparando features y variable objetivo...")

# Separación de variables independientes (X) y dependiente (y)
X = df.drop('Pago_atiempo', axis=1)  # Features: características del cliente
y = df['Pago_atiempo']               # Target: ¿Pagó a tiempo? (1=Sí, 0=No)

print(f"   Features: {X.shape[1]} variables predictoras")
print(f"   Target: {y.name}")


# ==============================================================================
# 5. DIVISIÓN ESTRATIFICADA: ENTRENAMIENTO Y PRUEBA
# ==============================================================================

print("\nPaso 3: Dividiendo datos en conjuntos de entrenamiento y prueba...")

# División estratificada para mantener proporción de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y  # Mantiene la proporción de clases
)

print(f"   Conjunto de entrenamiento: {X_train.shape[0]} muestras ({(1-TEST_SIZE)*100:.0f}%)")
print(f"   Conjunto de prueba: {X_test.shape[0]} muestras ({TEST_SIZE*100:.0f}%)")
print(f"   División estratificada aplicada para mantener balance de clases")


# ==============================================================================
# 6. FUNCIONES AUXILIARES PARA EVALUACIÓN
# ==============================================================================

def summarize_classification(y_true, y_pred, dataset_name="Dataset"):
    """
    Calcula un resumen completo de métricas de clasificación.
    
    PROPÓSITO DE CADA MÉTRICA:
    --------------------------
    - Accuracy: % de predicciones correctas (puede engañar con datos desbalanceados)
    - Precision: De los que predijimos "no pagarán", ¿cuántos acertamos?
                 (Evita alarmas falsas costosas)
    - Recall: De los que realmente no pagaron, ¿cuántos detectamos?
              (Evita dejar pasar clientes riesgosos - MÉTRICA CRÍTICA)
    - F1-Score: Balance armónico entre precision y recall
    - ROC-AUC: Capacidad del modelo para discriminar entre clases
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo
        dataset_name: Nombre del conjunto de datos (para logging)
    
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    # Cálculo de métricas estándar
    acc = accuracy_score(y_true, y_pred, normalize=True)
    prec = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    
    try:
        roc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # En caso de que solo haya una clase en y_true
        roc = 0.5
    
    # Conteo de casos críticos (clientes que NO pagarán a tiempo)
    casos_no_pago = np.count_nonzero(y_pred == POSITIVE_LABEL)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "casosNoPagoAtiempo": casos_no_pago
    }


def build_model(
    classifier_fn,
    data_params: dict,
    test_frac: float = 0.2,
    verbose: bool = True
) -> dict:
    """
    Pipeline completo de entrenamiento, validación y evaluación de modelos.
    
    FLUJO DEL PROCESO:
    ------------------
    1. Preprocesamiento de datos (feature engineering)
    2. Entrenamiento del modelo
    3. Validación cruzada (K-Fold) para medir generalización
    4. Curvas de aprendizaje para detectar overfitting/underfitting
    5. Análisis de escalabilidad (tiempo de entrenamiento vs datos)
    
    Args:
        classifier_fn: Instancia del modelo a entrenar
        data_params: Diccionario con información del dataset
        test_frac: Fracción de datos para prueba
        verbose: Si se muestran gráficos y logs detallados
    
    Returns:
        dict: Resultados de evaluación en train y test
    """
    
    # --- Extracción de parámetros ---
    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]
    
    # --- Preparación de datos ---
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]
    
    # División train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=RANDOM_STATE
    )
    
    # --- Feature Engineering ---
    # Aplicamos el pipeline de preprocesamiento personalizado
    preprocessor = ft_engineering(X_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    
    # --- Entrenamiento del modelo ---
    model_name = classifier_fn.__class__.__name__
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Entrenando modelo: {model_name}")
        print(f"{'='*80}")
    
    # Pipeline simple (el preprocesamiento ya se aplicó)
    classifier_pipe = Pipeline(steps=[("model", classifier_fn)])
    
    # Ajuste del modelo
    model = classifier_pipe.fit(x_train_processed, y_train)
    
    # --- Predicciones ---
    y_pred_train = model.predict(x_train_processed)
    y_pred_test = model.predict(x_test_processed)
    
    # --- Evaluación de desempeño ---
    train_summary = summarize_classification(y_train, y_pred_train, "Train")
    test_summary = summarize_classification(y_test, y_pred_test, "Test")
    
    if verbose:
        print(f"\nRESULTADOS DE {model_name}:")
        print(f"\n   TRAIN:")
        print(f"      Accuracy:  {train_summary['accuracy']:.4f}")
        print(f"      Precision: {train_summary['precision']:.4f}")
        print(f"      Recall:    {train_summary['recall']:.4f}")
        print(f"      F1-Score:  {train_summary['f1_score']:.4f}")
        print(f"      ROC-AUC:   {train_summary['roc_auc']:.4f}")
        
        print(f"\n   TEST:")
        print(f"      Accuracy:  {test_summary['accuracy']:.4f}")
        print(f"      Precision: {test_summary['precision']:.4f}")
        print(f"      Recall:    {test_summary['recall']:.4f}")
        print(f"      F1-Score:  {test_summary['f1_score']:.4f}")
        print(f"      ROC-AUC:   {test_summary['roc_auc']:.4f}")
        
        # INTERPRETACIÓN AUTOMÁTICA DE RESULTADOS
        print(f"\n   INTERPRETACIÓN:")
        
        # Análisis de overfitting
        gap_accuracy = train_summary['accuracy'] - test_summary['accuracy']
        if gap_accuracy > 0.1:
            print(f"      Posible OVERFITTING detectado (gap accuracy: {gap_accuracy:.2%})")
            print(f"         - El modelo memorizo patrones del entrenamiento")
            print(f"         - Considerar: regularizacion, mas datos, menos complejidad")
        elif gap_accuracy < 0.05:
            print(f"      Buena generalizacion (gap accuracy: {gap_accuracy:.2%})")
        
        # Análisis de recall (métrica crítica)
        if test_summary['recall'] < 0.5:
            print(f"      Recall bajo: solo detectamos {test_summary['recall']:.1%} de casos riesgosos")
            print(f"         - Muchos clientes problematicos pasaran desapercibidos")
        elif test_summary['recall'] > 0.8:
            print(f"      Excelente recall: detectamos {test_summary['recall']:.1%} de casos riesgosos")
        
        # Análisis de precision
        if test_summary['precision'] < 0.3:
            print(f"      Precision baja: muchas falsas alarmas")
            print(f"         - Podriamos rechazar clientes buenos innecesariamente")
    
    # --- Validación Cruzada ---
    if verbose:
        print(f"\nEjecutando validación cruzada ({N_CROSS_VALIDATION_FOLDS}-Fold)...")
    
    kfold = KFold(n_splits=N_CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    
    # Validación para cada métrica (excepto roc_auc por limitaciones)
    for metric in SCORING_METRICS[:-1]:
        cv_results[metric] = cross_val_score(
            classifier_pipe, x_train_processed, y_train, 
            cv=kfold, 
            scoring=metric
        )
    
    if verbose:
        print(f"   Validacion cruzada completada")
        for metric, scores in cv_results.items():
            print(f"      {metric.capitalize():12} - Media: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    # --- Curvas de Aprendizaje ---
    # Estas curvas nos ayudan a entender si necesitamos más datos o un modelo diferente
    if verbose:
        print(f"\nGenerando curvas de aprendizaje...")
    
    common_params = {
        "X": x_train_processed,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=N_LEARNING_CURVE_SPLITS, test_size=0.2, random_state=RANDOM_STATE),
        "n_jobs": -1,
        "return_times": True,
    }
    
    scoring_metric = "recall"  # Nuestra métrica más importante
    
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        classifier_pipe, **common_params, scoring=scoring_metric
    )
    
    # --- Cálculo de estadísticas ---
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)
    
    # --- VISUALIZACIÓN 1: Curva de Aprendizaje ---
    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
        
        # Líneas principales
        ax.plot(train_sizes, train_mean, "o-", linewidth=2, 
                label="Desempeño en Entrenamiento", markersize=8)
        ax.plot(train_sizes, test_mean, "o-", linewidth=2, 
                color="orange", label="Desempeño en Validación", markersize=8)
        
        # Areas de confianza (+/- 1 desviacion estandar)
        ax.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.15)
        ax.fill_between(train_sizes, test_mean - test_std, 
                        test_mean + test_std, alpha=0.15, color="orange")
        
        # Configuración visual
        ax.set_title(
            f"Curva de Aprendizaje: {model_name}\n"
            f"¿Más datos mejorarán el modelo?", 
            fontsize=14, fontweight='bold', pad=20
        )
        ax.set_xlabel("Número de ejemplos de entrenamiento", fontsize=12)
        ax.set_ylabel(f"{scoring_metric.capitalize()} (métrica objetivo)", fontsize=12)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Anotaciones interpretativas
        final_gap = train_mean[-1] - test_mean[-1]
        ax.text(
            0.02, 0.98, 
            f"Gap final (train-test): {final_gap:.3f}\n"
            f"Test final: {test_mean[-1]:.3f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )
        
        plt.tight_layout()
        plt.show()
        
        # Interpretacion de la curva
        print(f"\n   ANALISIS DE LA CURVA DE APRENDIZAJE:")
        
        if test_mean[-1] < test_mean[-2]:
            print(f"      - Curva de validacion estable o descendente")
            print(f"      - Mas datos podrian NO mejorar significativamente el modelo")
            print(f"      - Considerar: feature engineering o cambio de algoritmo")
        else:
            improvement = test_mean[-1] - test_mean[0]
            print(f"      - Mejora progresiva de {improvement:.2%} con mas datos")
            if final_gap > 0.15:
                print(f"      - El modelo aun puede beneficiarse de MAS DATOS")
            else:
                print(f"      - El modelo converge bien con los datos actuales")
    
    # --- VISUALIZACIÓN 2: Escalabilidad Computacional ---
    if verbose:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        
        # Subplot 1: Tiempo de entrenamiento
        ax[0].plot(train_sizes, fit_times_mean, "o-", linewidth=2, color="steelblue")
        ax[0].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.2, color="steelblue"
        )
        ax[0].set_ylabel("Tiempo de entrenamiento (segundos)", fontsize=11)
        ax[0].set_title(
            f"Escalabilidad Computacional: {model_name}\n"
            f"¿Cómo crece el tiempo de procesamiento?", 
            fontsize=14, fontweight='bold'
        )
        ax[0].grid(True, alpha=0.3)
        
        # Subplot 2: Tiempo de predicción
        ax[1].plot(train_sizes, score_times_mean, "o-", linewidth=2, color="coral")
        ax[1].fill_between(
            train_sizes,
            score_times_mean - score_times_std,
            score_times_mean + score_times_std,
            alpha=0.2, color="coral"
        )
        ax[1].set_ylabel("Tiempo de predicción (segundos)", fontsize=11)
        ax[1].set_xlabel("Número de ejemplos de entrenamiento", fontsize=11)
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analisis de escalabilidad
        print(f"\n   ANALISIS DE ESCALABILIDAD:")
        time_ratio = fit_times_mean[-1] / fit_times_mean[0]
        data_ratio = train_sizes[-1] / train_sizes[0]
        
        print(f"      Incremento de datos: {data_ratio:.1f}x")
        print(f"      Incremento de tiempo: {time_ratio:.1f}x")
        
        if time_ratio < data_ratio:
            print(f"      Escalabilidad SUB-LINEAL (eficiente)")
        elif time_ratio > data_ratio * 2:
            print(f"      Escalabilidad SUPER-LINEAL (puede ser lento con big data)")
        else:
            print(f"      - Escalabilidad LINEAL (proporcional)")
    
    return {"train": train_summary, "test": test_summary}


# ==============================================================================
# 7. DEFINICIÓN DE MODELOS A COMPARAR (ENSEMBLE APPROACH)
# ==============================================================================

print("\n" + "="*80)
print("FASE DE EXPERIMENTACIÓN: COMPARACIÓN DE MODELOS")
print("="*80)

print("""
ESTRATEGIA: Probaremos múltiples algoritmos con diferentes enfoques:

1. LOGISTIC REGRESSION: Modelo lineal simple, interpretable (baseline)
2. LINEAR SVC: Máxima separación entre clases (Support Vector Machine)
3. DECISION TREE: Reglas interpretables tipo "si-entonces"
4. RANDOM FOREST: Conjunto de árboles para reducir varianza
5. XGBOOST: Gradient boosting, estado del arte en tabular data

Todos configurados con class_weight='balanced' para manejar desbalanceo.
""")

# Diccionario de modelos con configuraciones optimizadas
models = {
    "logistic": LogisticRegression(
        solver="liblinear",
        class_weight='balanced',  # Penaliza más errores en clase minoritaria
        random_state=RANDOM_STATE
    ),
    
    "svc": LinearSVC(
        C=1.0,                    # Regularización moderada
        max_iter=1000,
        tol=1e-3,
        dual=False,               # Mejor para n_samples > n_features
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    
    "decision_tree": DecisionTreeClassifier(
        max_depth=5,              # Limita profundidad para evitar overfitting
        min_samples_leaf=10,      # Mínimo de muestras por hoja
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    
    "random_forest": RandomForestClassifier(
        n_estimators=150,         # 150 árboles en el bosque
        max_depth=7,              # Árboles no muy profundos
        min_samples_leaf=5,
        max_features='sqrt',      # Aleatorización en features
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1                 # Paralelización
    ),
    
    "xgboost": XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=491 / 10090,  # Balance basado en proporción de clases
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,            # 80% de muestras por árbol
        colsample_bytree=0.8,     # 80% de features por árbol
        random_state=RANDOM_STATE
    )
}

# Parámetros del dataset
data_params = {
    "name_of_y_col": y.name,
    "names_of_x_cols": X.columns,
    "dataset": df
}


# ==============================================================================
# 8. ENTRENAMIENTO Y EVALUACIÓN DE TODOS LOS MODELOS
# ==============================================================================

result_dict = {}

print(f"\nIniciando entrenamiento de {len(models)} modelos...")
print(f"   (Esto puede tomar varios minutos dependiendo del tamaño del dataset)\n")

for idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"\n[{idx}/{len(models)}] Procesando: {model_name.upper()}")
    print("-" * 80)
    
    try:
        result_dict[model_name] = build_model(
            model, 
            data_params, 
            test_frac=TEST_SIZE,
            verbose=True
        )
        print(f"{model_name.upper()} completado exitosamente")
        
    except Exception as e:
        print(f"Error entrenando {model_name}: {str(e)}")
        continue

print("\n" + "="*80)
print("FASE DE ENTRENAMIENTO COMPLETADA")
print("="*80)


# ==============================================================================
# 9. CONSOLIDACIÓN DE RESULTADOS
# ==============================================================================

print("\nConsolidando resultados de todos los modelos...\n")

# Transformar diccionario anidado a formato tabular
records = []
for model_name, model_results in result_dict.items():
    for data_set, metrics in model_results.items():
        for metric_name, score in metrics.items():
            records.append({
                "Model": model_name,
                "Data Set": data_set,
                "Metric": metric_name,
                "Score": score
            })

results_df = pd.DataFrame(records)

# Mostrar tabla resumida
print("="*80)
print("TABLA COMPARATIVA DE DESEMPEÑO")
print("="*80 + "\n")

# Pivot para visualización más clara
pivot_results = results_df.pivot_table(
    index=['Model', 'Metric'],
    columns='Data Set',
    values='Score'
)
print(pivot_results.to_string())
print("\n")

# ANÁLISIS AUTOMÁTICO: Identificar el mejor modelo para cada métrica
print("="*80)
print("MEJORES MODELOS POR MÉTRICA (en Test)")
print("="*80 + "\n")

test_results = results_df[results_df['Data Set'] == 'test']

for metric in ['recall', 'f1_score', 'precision', 'accuracy', 'roc_auc']:
    metric_data = test_results[test_results['Metric'] == metric]
    if not metric_data.empty:
        best_row = metric_data.loc[metric_data['Score'].idxmax()]
        print(f"   {metric.upper():12} -> {best_row['Model']:15} (score: {best_row['Score']:.4f})")

print("\n")


# ==============================================================================
# 10. VISUALIZACIÓN COMPARATIVA: STORYTELLING CON DATOS
# ==============================================================================

print("="*80)
print("GENERANDO VISUALIZACIONES COMPARATIVAS")
print("="*80 + "\n")

# Configuración de la figura
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
axes = axes.flatten()

# Título general de la narrativa
fig.suptitle(
    'COMPARACIÓN DE MODELOS: ¿Cuál predice mejor el riesgo de impago?\n'
    'Análisis Multi-dimensional del Desempeño Predictivo',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

# Métricas a visualizar
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

# Diccionario de contextos narrativos para cada métrica
metric_narratives = {
    "accuracy": {
        "title": "Accuracy General\n¿Qué % de predicciones son correctas?",
        "insight": "Útil como referencia general, pero puede engañar con clases desbalanceadas"
    },
    "precision": {
        "title": "Precisión: No Pago a Tiempo\n¿Cuántas alarmas son verdaderas?",
        "insight": "Alta precisión = Pocas falsas alarmas (rechazamos menos buenos clientes)"
    },
    "recall": {
        "title": "Recall: No Pago a Tiempo\n¿Detectamos a los clientes riesgosos?",
        "insight": "MÉTRICA CRÍTICA: Alto recall = Capturamos más casos problemáticos"
    },
    "f1_score": {
        "title": "F1-Score: No Pago a Tiempo\nBalance entre Precision y Recall",
        "insight": "Métrica balanceada, ideal cuando precision y recall son importantes"
    },
    "roc_auc": {
        "title": "ROC-AUC\n¿Qué tan bien separa clases el modelo?",
        "insight": "0.5 = aleatorio, 1.0 = perfecto. Bueno: >0.7, Excelente: >0.85"
    }
}

# Generar subplot para cada métrica
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    # Filtrar datos de la métrica actual
    metric_df = results_df[results_df["Metric"] == metric]
    
    # Gráfico de barras agrupado
    sns.barplot(
        data=metric_df, 
        x="Model", 
        y="Score", 
        hue="Data Set", 
        ax=ax, 
        palette={"train": "#3498db", "test": "#e74c3c"}
    )
    
    # Personalización del subplot
    narrative = metric_narratives.get(metric, {"title": metric, "insight": ""})
    
    ax.set_title(narrative["title"], fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Puntuación", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(title="Dataset", fontsize=10, title_fontsize=11)
    
    # Agregar línea de referencia para métricas donde aplique
    if metric in ['accuracy', 'roc_auc']:
        if metric == 'roc_auc':
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Aleatorio')
            ax.set_ylim(0.45, 1.05)
        else:
            ax.set_ylim(0, 1.05)
    elif metric == 'casosNoPagoAtiempo':
        max_no_pago = results_df[results_df["Metric"] == "casosNoPagoAtiempo"]["Score"].max()
        ax.set_ylim(0, max_no_pago + 5)
    else:
        ax.set_ylim(0, 1.05)
    
    # Agregar cuadro de texto con insight
    ax.text(
        0.5, -0.35, 
        narrative["insight"],
        transform=ax.transAxes,
        fontsize=9,
        style='italic',
        ha='center',
        wrap=True,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
    )
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# Ocultar el subplot vacío
axes[-1].set_visible(False)

# Ajustar layout
plt.tight_layout(rect=[0, 0.01, 1, 0.99])
plt.subplots_adjust(hspace=0.35, wspace=0.25)

print("   Visualización comparativa generada")

# Agregar panel de conclusiones en el espacio vacío
conclusion_ax = axes[-1]
conclusion_ax.set_visible(True)
conclusion_ax.axis('off')

# Identificar mejor modelo basado en recall (nuestra prioridad)
recall_test = test_results[test_results['Metric'] == 'recall']
if not recall_test.empty:
    best_model_row = recall_test.loc[recall_test['Score'].idxmax()]
    best_model = best_model_row['Model']
    best_recall = best_model_row['Score']
    
    # Obtener otras métricas del mejor modelo
    best_model_metrics = test_results[test_results['Model'] == best_model]
    
    conclusion_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║         RECOMENDACION ESTRATEGICA                   ║
    ╚══════════════════════════════════════════════════════╝
    
    MODELO RECOMENDADO: {best_model.upper()}
    
    POR QUE ESTE MODELO?
    - Recall mas alto: {best_recall:.1%}
    - Detecta la mayoria de clientes riesgosos
    - Balance adecuado entre metricas
    
    PROXIMOS PASOS:
    1. Validar con stakeholders el balance recall/precision
    2. Ajustar threshold segun tolerancia al riesgo
    3. Implementar en ambiente de produccion
    4. Monitorear drift y performance continuamente
    
    IMPACTO ESPERADO:
    - Reduccion de morosidad
    - Mejor gestion de riesgos
    - Decisiones de credito mas informadas
    """
    
    conclusion_ax.text(
        0.5, 0.5, 
        conclusion_text,
        transform=conclusion_ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2, pad=1)
    )

plt.show()

print("\nAnálisis visual completado")


# ==============================================================================
# 11. FUNCIÓN DE EVALUACIÓN PARA PRODUCCIÓN
# ==============================================================================

def evaluation():
    """
    Función de evaluación para modelos en producción.
    
    PROPÓSITO:
    ---------
    Esta función está diseñada para evaluar el modelo XGBoost guardado
    en producción, cargando el modelo serializado y generando reportes
    visuales para monitoreo continuo.
    
    FLUJO:
    ------
    1. Carga modelo guardado (.json)
    2. Prepara datos nuevos con feature engineering
    3. Genera predicciones
    4. Crea visualizaciones para stakeholders
    5. Retorna buffer de imagen para integración (API, reportes, dashboards)
    
    Returns:
    -------
    io.BytesIO: Buffer con imagen PNG del reporte de evaluación
    """
    
    print("\n" + "="*80)
    print("EVALUACIÓN DE MODELO EN PRODUCCIÓN")
    print("="*80 + "\n")
    
    # --- Cargar modelo guardado ---
    print("Cargando modelo XGBoost desde disco...")
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    model_features = model.feature_names
    print(f"   Modelo cargado: {len(model_features)} features")
    
    # --- Preparar datos ---
    print("\nAplicando feature engineering a nuevos datos...")
    df_new = ft_engineering()
    df_new.columns = [x.replace('__', '_') for x in df_new.columns]  # Normalizar nombres
    
    X_eval = df_new[model_features]
    X_eval_matrix = xgb.DMatrix(X_eval)
    Y_eval = df_new["Pago_atiempo"]
    
    print(f"   Datos preparados: {X_eval.shape[0]} registros")
    
    # --- Generar predicciones ---
    print("\nGenerando predicciones...")
    y_pred_proba = model.predict(X_eval_matrix)
    
    # Umbral de decisión (ajustable según tolerancia al riesgo)
    threshold = 0.5
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]
    
    print(f"   Predicciones generadas (threshold: {threshold})")
    print(f"   Casos predichos como 'No pago a tiempo': {sum(1 for p in y_pred if p == 0)}")
    
    # --- Calcular métricas ---
    eval_metrics = summarize_classification(Y_eval, y_pred, "Production")
    
    # --- Crear visualización de reporte ---
    print("\nGenerando reporte visual...")
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "casosNoPagoAtiempo"]
    
    # Preparar datos para visualización
    plot_data = []
    for metric in metrics_to_plot:
        if metric in eval_metrics:
            plot_data.append({
                "Model": "xgboost",
                "Data Set": "production",
                "Metric": metric,
                "Score": eval_metrics[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Crear figura
    fig, axes = plt.subplots(3, 2, figsize=(30, 18))
    axes = axes.flatten()
    
    fig.suptitle(
        'EVALUACIÓN DE MODELO XGBOOST EN PRODUCCIÓN\n'
        'Monitoreo de Desempeño y Métricas Clave',
        fontsize=30,
        fontweight='bold'
    )
    
    # Títulos personalizados para mejor narrativa
    custom_titles = {
        'precision': 'Precisión: No Pago a Tiempo\n¿Cuántas de nuestras alertas son correctas?',
        'recall': 'Recall: No Pago a Tiempo\n¿Estamos detectando los casos riesgosos?',
        'f1_score': 'F1-Score: No Pago a Tiempo\nBalance General del Modelo',
        'accuracy': 'Accuracy General\nPorcentaje de Aciertos Totales',
        'roc_auc': 'ROC-AUC\nCapacidad Discriminativa del Modelo',
        'casosNoPagoAtiempo': 'Casos Detectados: No Pago a Tiempo\nVolumen de Alertas Generadas'
    }
    
    # Generar subplots
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Filtrar datos de la métrica
        metric_df = plot_df[plot_df["Metric"] == metric]
        
        if not metric_df.empty:
            # Gráfico de barras
            sns.barplot(
                data=metric_df, 
                x="Model", 
                y="Score", 
                hue="Data Set", 
                ax=ax, 
                palette="rocket"
            )
            
            ax.legend(fontsize=18, title='Dataset', title_fontsize=20)
            
            title = custom_titles.get(metric, metric.replace("_", " ").title())
            ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
            ax.set_xticks([])
            ax.set_ylabel("Puntuación", fontsize=20, fontweight='bold')
            ax.set_xlabel("")
            ax.tick_params(axis='y', labelsize=18)
            
            # Ajustar límites
            if metric == 'roc_auc':
                ax.set_ylim(0.45, 1.05)
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label='Baseline Aleatorio')
            elif metric == 'casosNoPagoAtiempo':
                max_val = metric_df["Score"].max()
                ax.set_ylim(0, max_val * 1.2)
            else:
                ax.set_ylim(0, 1.05)
            
            # Agregar valor en la barra
            score_value = metric_df["Score"].values[0]
            ax.text(
                0, score_value, 
                f'{score_value:.3f}',
                ha='center', 
                va='bottom', 
                fontsize=16, 
                fontweight='bold'
            )
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ocultar subplot vacío si existe
    if len(metrics_to_plot) < len(axes):
        axes[-1].set_visible(False)
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar en buffer para retornar
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    print("   Reporte visual generado exitosamente")
    print("\n" + "="*80)
    
    return buf


# ==============================================================================
# 12. MENSAJE FINAL Y CONCLUSIONES
# ==============================================================================

print("\n" + "="*80)
print("PIPELINE DE MACHINE LEARNING COMPLETADO EXITOSAMENTE")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RESUMEN DEL PROCESO                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

COMPLETADO:
   - Carga y preparacion de datos
   - Feature engineering aplicado
   - Entrenamiento de 5 modelos diferentes
   - Validacion cruzada exhaustiva
   - Curvas de aprendizaje y escalabilidad
   - Visualizaciones comparativas
   - Identificacion del mejor modelo

MODELOS EVALUADOS:
   - Logistic Regression (baseline interpretable)
   - Linear SVC (maxima separacion)
   - Decision Tree (reglas claras)
   - Random Forest (ensemble robusto)
   - XGBoost (estado del arte)

ENFOQUE:
   - Prioridad en RECALL para detectar clientes riesgosos
   - Balance con PRECISION para evitar falsas alarmas
   - Analisis de overfitting y generalizacion
   - Consideracion de escalabilidad computacional

PROXIMOS PASOS RECOMENDADOS:
   1. Validar modelo seleccionado con equipo de negocio
   2. Ajustar threshold segun apetito de riesgo
   3. Implementar A/B testing en produccion
   4. Configurar monitoreo de drift
   5. Establecer proceso de re-entrenamiento periodico

RECUERDA:
   "Un modelo es tan bueno como los datos con los que se entrena
    y tan util como las decisiones que ayuda a tomar."

╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\nARCHIVOS GENERADOS:")
print("   - results_df: DataFrame con todas las metricas")
print("   - result_dict: Diccionario con resultados por modelo")
print("   - Visualizaciones: Graficos comparativos mostrados")

print("\nFUNCIONES DISPONIBLES:")
print("   - summarize_classification(): Calcular metricas de un modelo")
print("   - build_model(): Entrenar y evaluar un nuevo modelo")
print("   - evaluation(): Evaluar modelo XGBoost en produccion")

print("\n" + "="*80)
print("Fin del script - ¡Feliz modelado!")
print("="*80 + "\n")
