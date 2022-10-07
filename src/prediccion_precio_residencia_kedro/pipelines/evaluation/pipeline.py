"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.evaluation.nodes import evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluation,
            inputs=['y_real_train_test', 'y_real_validation', 'y_predict_train_test', 'y_predict_validation'],
            outputs=['score_train_test', 'score_validation']
        )
    ])
