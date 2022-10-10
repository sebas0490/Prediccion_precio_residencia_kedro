"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.3
"""
from typing import Dict, Any

from pandas import DataFrame

from prediccion_precio_residencia_kedro.models.modelo import Modelo

import mlflow


def train_model(
        df_train_test_transformed: DataFrame,
        parameters: Dict[str, Any]
) -> Modelo:
    modelo = Modelo(parameters['columnas_entrada'])
    modelo.fit(df_train_test_transformed[parameters['X_columns']], df_train_test_transformed[parameters['y_column']])
    mlflow.set_tag("mlflow.runName", modelo.__class__.__name__)
    return modelo
