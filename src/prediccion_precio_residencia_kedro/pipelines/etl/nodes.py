"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from pandas import DataFrame


def get_data() -> DataFrame:
    data = DataFrame({'x': list(range(100)), 'y': list(map(lambda x: x*2, range(100)))})
    return data
