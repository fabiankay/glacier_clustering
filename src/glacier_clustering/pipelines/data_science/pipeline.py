"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import encode_data, scale_data, train_model, visualize_model, create_model_data


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline([
        node(
            func=encode_data,
            inputs=["merged_data", "params:model_options"],
            outputs=["encoded_data", "encoder"],
            name="encode_data_node",
        ),
        node(
            func=scale_data,
            inputs=["merged_data", "params:model_options"],
            outputs=["scaled_data", "scaler"],
            name="scale_data_node",
        ),
        node(
            func=create_model_data,
            inputs=["encoded_data", "scaled_data"],
            outputs="model_input_table",
            name="create_model_data_node",
        ),
        node(
            func=train_model,
            inputs=["model_input_table", "params:model_options"],
            outputs=["model", "labels", "centers"],
            name="train_model_node",
        ),
        node(
            func=visualize_model,
            inputs=["merged_data", "labels"],
            outputs="cluster_map",
            name="visualize_model_node",
        ),
    ])

    ds_pipeline_1 = pipeline(
        pipe=pipeline_instance,
        inputs="merged_data",
        namespace="active_modelling_pipeline",
    )
    ds_pipeline_2 = pipeline(
        pipe=pipeline_instance,
        inputs="merged_data",
        namespace="candidate_modelling_pipeline",
    )

    return ds_pipeline_1 + ds_pipeline_2
