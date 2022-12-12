"""
This is a boilerplate pipeline 'merge_data'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import merge_data, load_state, load_change, load_glacier, load_mass_balance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_glacier,
            inputs="glacier",
            outputs="loaded_glacier",
            name="load_glacier_node",
        ),
        node(
            func=load_change,
            inputs="change",
            outputs="loaded_change",
            name="load_change_node",
        ),
        node(
            func=load_state,
            inputs="state",
            outputs="loaded_state",
            name="load_state_node",
        ),
        node(
            func=load_mass_balance,
            inputs="mass_balance",
            outputs="loaded_mass_balance",
            name="load_mass_balance_node",
        ),
        node(
            func=merge_data,
            inputs=["loaded_glacier", "loaded_change", "loaded_state", "loaded_mass_balance"],
            outputs="merged_data",
            name="merge_data_node",
        )
    ])
