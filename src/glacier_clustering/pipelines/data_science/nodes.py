import logging
from typing import Tuple, Dict, List

import pandas as pd
from plotly.graph_objs import Figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import plotly.express as px


def scale_data(X: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, StandardScaler]:
    """Trains the linear regression model.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled data of independent features and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[parameters["num_features"]])
    return X_scaled, scaler


def encode_data(X: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encodes categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.

    Returns
    -------
    Tuple[pd.DataFrame, OneHotEncoder]
        Encoded data of independent features and fitted encoder.
    """
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X[parameters["cat_features"]])
    return X_encoded, encoder


def create_model_data(X_scaled: pd.DataFrame, X_encoded: pd.DataFrame) -> pd.DataFrame:
    """Creates a table with the model input data.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Data of scaled features.

    X_encoded : pd.DataFrame
        Data of encoded features.

    Returns
    -------
    pd.DataFrame
        Table with the model input data.
    """
    return pd.concat([
        pd.DataFrame(X_scaled.toarray()),
        pd.DataFrame(X_encoded)
    ], axis=1)


def train_model(X: pd.DataFrame, parameters: Dict) -> Tuple[object, List, List]:
    """Trains a KMeans model.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.

    Returns
    -------
    KMeans
        Trained model.
    """
    kmeans = KMeans(
        n_clusters=parameters["n_clusters"],
        max_iter=parameters["max_iter"],
        tol=parameters["tol"],
        random_state=parameters["random_state"]
    ).fit(X)
    return kmeans, kmeans.labels_, kmeans.cluster_centers_


def visualize_model(X: pd.DataFrame, labels: List) -> Figure:
    """Visualizes the model by plotting the clusters.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.
    model : KMeans
        Trained model.

    Returns
    -------
    None
    """
    data_names = ["PRIM_CLASSIFIC", "FORM", "FRONTAL_CHARS", "AREA", "LENGTH", "HIGHEST_ELEVATION",
                  "MEDIAN_ELEVATION", "LOWEST_ELEVATION", "AREA_CHANGE", "THICKNESS_CHG", "VOLUME_CHANGE",
                  "WINTER_BALANCE", "SUMMER_BALANCE", "ANNUAL_BALANCE"]
    fig = px.scatter_geo(X, lat="LATITUDE", lon="LONGITUDE", color=labels,
                     hover_name="WGMS_ID", projection="natural earth", title="Glacier Clusters", hover_data=data_names)
    return fig
