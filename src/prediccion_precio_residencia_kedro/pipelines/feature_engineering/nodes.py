"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""
from typing import Tuple, Dict, Any

from pandas import DataFrame

from prediccion_precio_residencia_kedro.data.procesamiento_datos import ProcesamientoDatos, Preprocesamiento


def train_transformers(df_train_test: DataFrame, parameters: Dict[str, Any]):
    """
    Returns: processing, preprocessing
    """
    processing = ProcesamientoDatos()
    processing.fit_transform(df_train_test)
    preprocessing = Preprocesamiento(parameters['columnas_z_score'], [])
    return processing, preprocessing


def feature_engineering(df_train_test: DataFrame,
                        df_validation: DataFrame,
                        processing: ProcesamientoDatos,
                        preprocessing: Preprocesamiento) -> Tuple[DataFrame, DataFrame]:
    """
    :return: df_train_test_transformed, df_validation_transformed
    """

    df_train_test_transformed = df_train_test. \
        pipe(preprocessing.transform). \
        pipe(processing.transform)
    df_validation_transformed = df_validation. \
        pipe(preprocessing.transform). \
        pipe(processing.transform)
    return df_train_test_transformed, df_validation_transformed
