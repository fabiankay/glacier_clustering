import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from plotly.graph_objs import Figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def scale_data(X: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scales the data.

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


def scale_timeseries(X: np.array, parameters: Dict) -> Tuple[np.array, TimeSeriesScalerMeanVariance]:
    """Scales the timeseries data.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled data of independent features and fitted scaler.
    """
    # np.random.shuffle(X)
    # Keep only 100 time series
    # print(X.shape)
    scaler = TimeSeriesScalerMeanVariance()
    X = scaler.fit_transform(X)
    return X, scaler


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


def train_model(X: np.array, parameters: Dict) -> Tuple[object, List, List]:
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

    km = TimeSeriesKMeans(
        n_clusters=parameters["n_clusters"],
        verbose=True,
        metric=parameters["metric"],
        random_state=parameters["random_state"],
        n_jobs=parameters["n_jobs"]
    )
    km.fit_predict(X)

    return km, km.labels_, km.cluster_centers_


def visualize_model(X: pd.DataFrame, labels: List, km: object, parameters: Dict):
    """Visualizes the model by plotting the clusters.

    Parameters
    ----------
    X : pd.DataFrame
        Data of independent features.
    labels : List
        List of cluster labels.
    km : KMeans
        Trained model.
    parameters : Dict
        Parameters of the pipeline.

    Returns
    -------
    None
    """

    sz = X.shape[1]
    fig = plt.figure()
    for yi in range(parameters["n_clusters"]):
        plt.subplot(3, 3, yi + 1)
        for xx in X[labels == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("$k$-means")

    return plt
