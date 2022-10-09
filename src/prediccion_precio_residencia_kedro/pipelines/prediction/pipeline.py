"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.prediction.nodes import prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prediction,
            inputs=['df_train_test_transformed', 'df_validation_transformed', 'trained_model', 'parameters'],
            outputs=['y_real_train_test', 'y_real_validation', 'y_predict_train_test', 'y_predict_validation'],
            tags='Model evaluation'
        )
    ])
