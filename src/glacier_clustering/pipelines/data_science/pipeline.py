"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_model, visualize_model, scale_timeseries


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scale_timeseries,
            inputs=["timeseries_data", "params:model_options"],
            outputs=["scaled_data", "scaler"],
            name="scale_data_node",
        ),
        node(
            func=train_model,
            inputs=["scaled_data", "params:model_options"],
            outputs=["model", "labels", "centers"],
            name="train_model_node",
        ),
        node(
            func=visualize_model,
            inputs=["reference_data", "labels", "model", "params:model_options"],
            outputs=["cluster_map", "cluster_data"],
            name="visualize_model_node",
            tags="visualize",
        ),
    ])
