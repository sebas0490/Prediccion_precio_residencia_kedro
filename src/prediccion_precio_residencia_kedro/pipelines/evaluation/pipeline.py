"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.evaluation.nodes import evaluation, model_evaluation_check


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluation,
            inputs=['y_real_train_test', 'y_real_validation', 'y_predict_train_test', 'y_predict_validation'],
            outputs=['score_train_test', 'score_validation'],
            tags='Model evaluation'
        ),
        node(
            func=model_evaluation_check,
            inputs=["df_train_test_transformed",
                    "df_validation_transformed",
                    "trained_model",
                    "parameters"
                    ],
            outputs=None,
            name='model_evaluation_check',
            tags='deepchecks'
        )
    ])
