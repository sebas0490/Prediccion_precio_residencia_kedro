"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""
import logging
import tempfile
from typing import Tuple, Dict, Any

import mlflow
import zipfile
import numpy as np
import pandas as pd
from deepchecks import Dataset
from pathlib import Path
from deepchecks.tabular.suites import data_integrity
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prediccion_precio_residencia_kedro.data.funciones_base import convertir_tipos, reemplazar_valores_extremos, \
    reemplazar_nulos_por_la_media, eliminar_duplicados, convertir_col_date_a_date, reemplazar_fechas_nulas, \
    reemplazar_ceros_por_nulos, calculo_variables_adicionales, validar_index_duplicados, seleccionar_columnas
from prediccion_precio_residencia_kedro.data.procesamiento_datos import Preprocesamiento
from prediccion_precio_residencia_kedro.jutils.data import DataUtils, DataAccess

logger = logging.getLogger(__name__)


def _convertir_tipos(df: DataFrame, columnas_numericas) -> DataFrame:
    return convertir_tipos(df, columnas_numericas)


def _reemplazar_valores_extremos(df: DataFrame, columnas_numericas) -> DataFrame:
    return reemplazar_valores_extremos(df, columnas_numericas)


def _reemplazar_nulos_por_la_media(df: DataFrame, columnas_numericas) -> DataFrame:
    return reemplazar_nulos_por_la_media(df, columnas_numericas)


def get_data(parameters: Dict[str, Any]) -> DataFrame:
    """
    Returns:
        data_raw
    """
    folder_path = 'cache'
    url = parameters['url']
    du = DataUtils(
        Path(folder_path + '/data').resolve().absolute(),
        'kc_house_dataDS.parquet',
        'price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda df, path: df.to_parquet(path)
    )

    def process_raw_file(path_to_downloaded_file: Path):
        with zipfile.ZipFile(path_to_downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(path_to_downloaded_file.parent)
        path_to_processed_file = path_to_downloaded_file.parent.joinpath('HouseKing/kc_house_dataDS.csv')
        return path_to_processed_file

    da = DataAccess(url, du, lambda path: pd.read_csv(path, sep=',', index_col=0),
                    process_raw_file)

    data_raw: DataFrame = da.get_df()
    mlflow.set_experiment('prediccion_casas')
    mlflow.log_param('data_raw_shape', data_raw.shape)
    return data_raw


def limpieza_calidad(data_raw: DataFrame,
                     parameters: Dict[str, Any]
                     ) -> DataFrame:
    columnas_numericas = list(set(parameters['columnas_raw']).difference(['date']))
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
    if 'index' not in data_raw.columns:
        cant_filas = data_raw.shape[0]
        data_raw['index'] = np.linspace(1, cant_filas, cant_filas)
    data_limpia = li.transform(data_raw)
    mlflow.log_param('shape_limpieza_datos', data_limpia.shape)
    return data_limpia


def eliminar_nulos_entrenamiento_validacion(
        data_transformada: DataFrame,
        parameters: Dict[str, Any]
) -> DataFrame:
    """
    Returns: data_transformada_validacion
    """
    # Cuando se inicialice en modo entrenamiento o validación se deben eliminar los registros con precios nulos
    #   o por fuera de lo normal de acuerdo con el z-score.
    if parameters['modo_entrenamiento_validacion']:
        pval = Preprocesamiento([], ['price'], parameters['columnas_entrada'])
        data_transformada_validacion = pval.fit_transform(data_transformada)
    else:
        data_transformada_validacion = data_transformada
    mlflow.log_param('shape data eliminar nulos', data_transformada_validacion.shape)
    return data_transformada_validacion


def make_dataset(parameters: Dict[str, Any],
                 data_transformada_validacion: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Returns: df_train_test, df_validation
    """
    porcentaje_entrenamiento = parameters['porcentaje_entrenamiento']
    if porcentaje_entrenamiento < 1:
        df_train_test, df_validation = \
            train_test_split(data_transformada_validacion, train_size=porcentaje_entrenamiento, random_state=1)
    else:
        df_train_test = data_transformada_validacion
        df_validation = data_transformada_validacion[[False] * data_transformada_validacion.shape[0]]
    mlflow.log_param('df_train_test_shape', df_train_test.shape)
    mlflow.log_param('df_validation_shape', df_validation.shape)
    return df_train_test, df_validation


def data_integrity_validation(data: pd.DataFrame,
                              parameters: Dict) -> pd.DataFrame:
    categorical_features = parameters['categorical_cols']
    label = parameters['y_column']

    dataset = Dataset(data,
                      label=label,
                      cat_features=categorical_features)
    mlflow.set_experiment('prediccion_casas')
    mlflow.log_param('X_columns', parameters['X_columns'])
    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(dataset)
    mlflow.log_param(f"data integrity validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        # mlflow.log_text('Se corre el modelo quitando los duplicados exactos', 'texto/nota2.txt')
        ruta = Path('data/08_reporting/data_integrity_check.html')
        ruta.unlink(missing_ok=True)
        suite_result.save_as_html(str(ruta))
        # mlflow.log_artifact(str(ruta), 'deepchecks')
        ruta.unlink(missing_ok=True)
        logger.error("data integrity not pass validation tests")
        # raise Exception("data integrity not pass validation tests")
    return data
