"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.feature_engineering.nodes import train_transformers, \
    feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_transformers,
            inputs=['df_train_test', 'parameters'],
            outputs=['processing', 'preprocessing']
        ),
        node(
            func=feature_engineering,
            inputs=['df_train_test', 'df_validation', 'processing', 'preprocessing'],
            outputs=['df_train_test_transformed', 'df_validation_transformed']
        )
    ])
