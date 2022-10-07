"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from prediccion_precio_residencia_kedro.data.procesamiento_datos import Preprocesamiento, ProcesamientoDatos


def build_preprocessing() -> Preprocesamiento:
    pass


def build_processing() -> ProcesamientoDatos:
    pass


def feature_engineering(df_train_test,
                        df_validation,
                        preprocsessing: Preprocesamiento,
                        processing: ProcesamientoDatos):
    df_train_test_transformed = df_train_test. \
        pipe(preprocsessing.fit_transform). \
        pipe(processing.fit_transform)
    df_validation_transformed = df_validation. \
        pipe(preprocsessing.fit_transform). \
        pipe(processing.fit_transform)
    return df_train_test_transformed, df_validation_transformed
