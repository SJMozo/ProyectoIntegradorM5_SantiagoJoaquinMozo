"""
Script para la carga de datos desde archivos Excel.

Este script proporciona funcionalidades para cargar datos desde archivos Excel
ubicados en el directorio del proyecto, facilitando el acceso a bases de datos
en formato .xlsx.
"""

import os
import pandas as pd

# Constantes del script
Archivo_excel = "C:\\Users\\Santi\\OneDrive\\Desktop\\avances modulo 5\\avances 2 y 3\\Base_de_datos.xlsx"


def cargar_datos(nombre_archivo: str = Archivo_excel) -> pd.DataFrame:
    """
    Carga datos desde un archivo Excel ubicado en el directorio padre del script.
    
    Esta función:
    1. Identifica el directorio actual del script
    2. Navega al directorio padre (proyecto raíz)
    3. Construye la ruta completa al archivo Excel
    4. Lee y retorna los datos como DataFrame de pandas
    
    Args:
        nombre_archivo (str): Nombre del archivo Excel a cargar.
                            Por defecto: 'Base_de_datos.xlsx'
    
    Returns:
        pd.DataFrame: DataFrame de pandas con los datos del archivo Excel.
    
    Raises:
        FileNotFoundError: Si el archivo Excel no existe en la ruta esperada.
        ValueError: Si el archivo existe pero no se puede leer correctamente.
        
    Example:
        >>> df = cargar_datos()
        >>> print(df.head())
    """
    try:
        # Paso 1: Obtener la ruta absoluta del directorio donde está este script
        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        
        # Paso 2: Navegar al directorio padre (raíz del proyecto)
        # Esto permite que el script funcione desde cualquier subdirectorio
        ruta_proyecto = os.path.dirname(ruta_actual)
        
        # Paso 3: Construir la ruta completa al archivo Excel
        # Utiliza os.path.join para garantizar compatibilidad multiplataforma
        ruta_excel = os.path.join(ruta_proyecto, nombre_archivo)
        
        # Verificar que el archivo existe antes de intentar leerlo
        if not os.path.exists(ruta_excel):
            raise FileNotFoundError(
                f"No se encontró el archivo '{nombre_archivo}' en: {ruta_proyecto}"
            )
        
        # Paso 4: Leer el archivo Excel y cargarlo en un DataFrame
        df = pd.read_excel(ruta_excel)
        
        # Mostrar el DataFrame completo para verificación
        print(df)
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        raise ValueError(
            f"Error al leer el archivo Excel '{nombre_archivo}': {str(e)}"
        ) from e


def main() -> None:
    """
    Función principal para ejecutar el módulo como script independiente.
    
    Demuestra el uso de la función cargar_datos() mostrando:
    - Las primeras filas del DataFrame
    - Los nombres de las columnas disponibles
    """
    # Cargar los datos del archivo Excel
    datos = cargar_datos()
    
    # Mostrar las primeras filas para una vista previa
    print("\n--- Primeras filas del DataFrame ---")
    print(datos.head())
    
    # Mostrar las columnas disponibles en el DataFrame
    print("\n--- Columnas disponibles ---")
    print(datos.columns)


# Punto de entrada cuando el script se ejecuta directamente
if __name__ == "__main__":
    main()