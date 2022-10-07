"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""
from typing import Tuple, Dict, Any, List

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prediccion_precio_residencia_kedro.core.steps import Steps
import tempfile

from prediccion_precio_residencia_kedro.data.funciones_base import convertir_tipos, reemplazar_valores_extremos, \
    reemplazar_nulos_por_la_media, eliminar_duplicados, convertir_col_date_a_date, reemplazar_fechas_nulas, \
    reemplazar_ceros_por_nulos, calculo_variables_adicionales, validar_index_duplicados, seleccionar_columnas
from prediccion_precio_residencia_kedro.data.procesamiento_datos import Preprocesamiento


def _convertir_tipos(df: DataFrame, columnas_numericas) -> DataFrame:
    return convertir_tipos(df, columnas_numericas)


def _reemplazar_valores_extremos(df: DataFrame, columnas_numericas) -> DataFrame:
    return reemplazar_valores_extremos(df, columnas_numericas)


def _reemplazar_nulos_por_la_media(df: DataFrame, columnas_numericas) -> DataFrame:
    return reemplazar_nulos_por_la_media(df, columnas_numericas)


def get_data(parameters: Dict[str, Any]) -> Tuple[DataFrame, Steps, List[str]]:
    """
    Returns:
        data, steps, columnas_numericas
    """
    steps = Steps.build(tempfile.gettempdir() + '/prediccion_precio_residencia_kedro', url=parameters['url'])
    data: DataFrame = steps.raw_df
    columnas_numericas = list(set(parameters['columnas_raw']).difference(['date']))
    return data, steps, columnas_numericas


def limpieza_calidad(data: DataFrame,
                     columnas_numericas: List[str],
                     parameters: Dict[str, Any]
                     ) -> Tuple[DataFrame, Steps]:
    li = Pipeline([
        ('convertir_tipos', FunctionTransformer(_convertir_tipos,
                                                kw_args={'columnas_numericas': columnas_numericas})),
        ('eliminar_duplicados', FunctionTransformer(eliminar_duplicados)),
        ('col_date_a_date', FunctionTransformer(convertir_col_date_a_date)),
        ('valores_extremos', FunctionTransformer(_reemplazar_valores_extremos,
                                                 kw_args={'columnas_numericas': columnas_numericas})),
        ('nulos_por_media', FunctionTransformer(_reemplazar_nulos_por_la_media,
                                                kw_args={'columnas_numericas': columnas_numericas})),
        ('fechas_nulas', FunctionTransformer(reemplazar_fechas_nulas)),
        ('ceros_por_nulos', FunctionTransformer(reemplazar_ceros_por_nulos)),
        ('eliminar_duplicados2', FunctionTransformer(eliminar_duplicados)),
        ('calculo_variables_adicionales', FunctionTransformer(calculo_variables_adicionales)),
        ('validar_indices_duplicados', FunctionTransformer(validar_index_duplicados,
                                                           kw_args={'is_duplicated_ok': False})),
        ('seleccionar_columnas', FunctionTransformer(seleccionar_columnas,
                                                     kw_args={'columnas': parameters['columnas_entrada'] + ['price']}))
    ]
    )
    if 'index' not in data.columns:
        cant_filas = data.shape[0]
        data['index'] = np.linspace(1, cant_filas, cant_filas)
    data_transformada = li.transform(data)
    return data_transformada


def eliminar_nulos_entrenamiento_validacion(data_transformada: DataFrame) -> Tuple[DataFrame, Steps]:
    """
    Returns: data_transformada_validacion
    """
    # Cuando se inicialice en modo entrenamiento o validación se deben eliminar los registros con precios nulos
    #   o por fuera de lo normal de acuerdo con el z-score.
    pval = Preprocesamiento([], ['price'])
    data_transformada = pval.fit_transform(data_transformada)
    return data_transformada


def make_dataset(parameters: Dict[str, Any], data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Returns: df_train_test, df_validation
    """
    porcentaje_entrenamiento = parameters['porcentaje_entrenamiento']
    if porcentaje_entrenamiento < 1:
        df_train_test, df_validation = train_test_split(data, train_size=porcentaje_entrenamiento, random_state=1)
    else:
        df_train_test = data
        df_validation = data[[False] * data.shape[0]]

    return df_train_test, df_validation
