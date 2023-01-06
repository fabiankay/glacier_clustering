import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
import seaborn as sns
from plotly.graph_objs import Figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


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
        n_jobs=parameters["n_jobs"],
        max_iter=parameters["max_iter"]
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

    reference_data = pd.DataFrame()

    # add labels to md by first index level
    for i, (idx, row) in enumerate(X.groupby(level=0)):
        row["LABEL"] = labels[i]
        reference_data = pd.concat([reference_data, row])

    sns.lineplot(data=reference_data.reset_index(), x="YEAR", y="THICKNESS_CHG", hue="LABEL")

    return plt, reference_data
