"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""
from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd
from deepchecks import Dataset
from deepchecks.tabular.suites import model_evaluation
from sklearn.metrics import r2_score

from prediccion_precio_residencia_kedro.models.modelo import Modelo


def evaluation(y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation):
    """
    :return: score_train_test, score_validation
    """
    score_train_test = r2_score(y_real_train_test, y_predict_train_test)
    score_validation = r2_score(y_real_validation, y_predict_validation)
    mlflow.log_metric('score_train_test', score_train_test)
    mlflow.log_metric('score_validation', score_validation)
    return f'{score_train_test=:.2%}', f'{score_validation=:.2%}'


def model_evaluation_check(
        df_train_test_transformed: pd.DataFrame,
        df_validation_transformed: pd.DataFrame,
        trained_model: Modelo,
        parameters: Dict[str, Any]):
    y_train_test = df_train_test_transformed[parameters['y_column']]
    x_train_test = df_train_test_transformed[parameters['X_columns']]
    y_validation = df_validation_transformed[parameters['y_column']]
    x_validation = df_validation_transformed[parameters['X_columns']]
    train_ds = Dataset(x_train_test, label=y_train_test, cat_features=[])
    test_ds = Dataset(x_validation, label=y_validation, cat_features=[])

    evaluation_suite = model_evaluation()
    suite_result = evaluation_suite.run(train_ds, test_ds, trained_model)
    mlflow.set_experiment('prediccion_casas')
    mlflow.log_param(f"model evaluation validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        ruta = Path('data/08_reporting/model_eval_check.html')
        ruta.unlink(missing_ok=True)
        suite_result.save_as_html(str(ruta))
        mlflow.log_artifact(str(ruta), 'deepchecks')
        ruta.unlink(missing_ok=True)
        print("model not pass evaluation tests")
