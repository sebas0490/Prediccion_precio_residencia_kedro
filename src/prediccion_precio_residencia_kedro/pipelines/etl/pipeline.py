"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_data,
            inputs=[],
            outputs='kc_house_dataDS',
            name='data_raw_download'
        )
    ])
