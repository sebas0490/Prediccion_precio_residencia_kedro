"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.training.nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=['df_train_test_transformed', 'parameters'],
            outputs='trained_model'
        )
    ])
