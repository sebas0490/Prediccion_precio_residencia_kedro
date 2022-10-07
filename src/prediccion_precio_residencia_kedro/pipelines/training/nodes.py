"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.3
"""
from typing import Tuple, Dict, Any

from kedro.pipeline import Pipeline
from pandas import DataFrame

from prediccion_precio_residencia_kedro.models.modelo import Modelo


def train_model(
        df_train_test_transformed: DataFrame,
        parameters: Dict[str, Any]
) -> Modelo:
    modelo = Modelo()
    modelo.fit(df_train_test_transformed[parameters['X_columns']], df_train_test_transformed[parameters['y_column']])
    return modelo
