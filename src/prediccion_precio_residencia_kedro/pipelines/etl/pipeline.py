"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.etl.nodes import get_data, limpieza_calidad, make_dataset, \
    eliminar_nulos_entrenamiento_validacion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_data,
            inputs='parameters',
            outputs='data_raw',
            name='data_raw_download'
        ),
        node(
            func=limpieza_calidad,
            inputs=['data_raw', 'parameters'],
            outputs='data_limpia',
            name='datos_limpios'
        ),
        node(
            func=eliminar_nulos_entrenamiento_validacion,
            inputs=['data_limpia', 'parameters'],
            outputs='data_validacion'
        ),
        node(
            func=make_dataset,
            inputs=['parameters', 'data_validacion'],
            outputs=['df_train_test', 'df_validation']
        )
    ])

