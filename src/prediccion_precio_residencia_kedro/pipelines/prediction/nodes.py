"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.3
"""
from typing import Dict, Any

from pandas import DataFrame

from prediccion_precio_residencia_kedro.models.modelo import Modelo


def prediction(
        df_train_test_transformed: DataFrame,
        df_validation_transformed: DataFrame,
        modelo: Modelo,
        parameters: Dict[str, Any]
):
    """
    :return: y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation
    """
    y_real_train_test = df_train_test_transformed[parameters['y_column']]
    y_real_validation = df_validation_transformed[parameters['y_column']]
    y_predict_train_test = modelo.predict(df_train_test_transformed)
    y_predict_validation = modelo.predict(df_validation_transformed)
    return y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation
