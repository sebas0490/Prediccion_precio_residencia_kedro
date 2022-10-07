"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_engineering, build_preprocessing, build_processing
from ...data.procesamiento_datos import Preprocesamiento, ProcesamientoDatos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_preprocessing,
            inputs=None,
            outputs='preprocessing'
        ),
        node(
            func=build_processing,
            inputs=None,
            outputs='processing'
        ),
        node(
            func=feature_engineering,
            inputs=['df_train_test', 'df_validation', 'preprocessing', 'processing'],
            outputs=['df_train_test_transformed', 'df_validation_transformed']
        )
    ])
