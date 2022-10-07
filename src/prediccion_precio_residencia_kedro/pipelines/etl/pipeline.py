"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from prediccion_precio_residencia_kedro.pipelines.etl.nodes import get_data, limpieza_calidad, make_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_data,
            inputs='parameters',
            outputs=['kc_house_dataDS', 'steps', 'columnas_numericas'],
            name='data_raw_download'
        ),
        node(
            func=limpieza_calidad,
            inputs=['kc_house_dataDS', 'columnas_numericas', 'parameters'],
            outputs='dataset_limpio',
            name='datos_limpios'
        ),
        node(
            func=make_dataset,
            inputs=['parameters', 'dataset_limpio'],
            outputs=['df_train_test', 'df_validation']
        )
    ])

