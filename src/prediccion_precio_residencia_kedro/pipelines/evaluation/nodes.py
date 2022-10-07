"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""
from sklearn.metrics import r2_score


def evaluation(y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation):
    """
    :return: score_train_test, score_validation
    """
    score_train_test = r2_score(y_real_train_test, y_predict_train_test)
    score_validation = r2_score(y_real_validation, y_predict_validation)
    return f'{score_train_test=:.2%}', f'{score_validation=:.2%}'
