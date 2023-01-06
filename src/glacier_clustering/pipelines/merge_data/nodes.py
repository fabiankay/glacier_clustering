from typing import Tuple, Dict

import numpy as np
import pandas as pd
from tslearn.utils import to_time_series_dataset
from logging import getLogger

log = getLogger(__name__)


def get_cumsum(df):
    cols = df.select_dtypes(include=np.number).columns.tolist()
    df = df.loc[df["YEAR"] >= 2000, cols].sort_values(["YEAR", "WGMS_ID"])
    return df.groupby(["WGMS_ID", "YEAR"]).sum().groupby(level=0).cumsum().reset_index()


def load_glacier(glacier: pd.DataFrame) -> pd.DataFrame:
    """Load glaciers to remove duplicates and glaciers without position.

    Parameters
    ----------
    glacier : pd.DataFrame
        Glaciers to preprocess.

    Returns
    -------
    pd.DataFrame
        Preprocessed glaciers.
    """

    glacier = glacier.drop_duplicates(subset=["WGMS_ID"])
    glacier = glacier.dropna(subset=["WGMS_ID", "LATITUDE", "LONGITUDE"])
    # fill nan values with 99 for "PRIM_CLASSIFIC", "FORM", "FRONTAL_CHARS"
    glacier = glacier.fillna({"PRIM_CLASSIFIC": 99, "FORM": 99, "FRONTAL_CHARS": 99})
    glacier = glacier[["WGMS_ID", "LATITUDE", "LONGITUDE", "PRIM_CLASSIFIC", "FORM", "FRONTAL_CHARS"]]
    return glacier


def load_mass_balance(mass_balance: pd.DataFrame) -> pd.DataFrame:
    """ Load and preprocess mass balance data.

    Parameters
    ----------
    mass_balance : pd.DataFrame
        Mass balance data.

    Returns
    -------
    pd.DataFrame
        Preprocessed mass balance data.
    """

    # mass_balance = mass_balance.drop_duplicates(subset=["WGMS_ID"])
    mass_balance = mass_balance.dropna(subset=["WGMS_ID"])
    mass_balance = mass_balance.loc[
        mass_balance["YEAR"] >= 2000,
        ["WGMS_ID", "YEAR", "WINTER_BALANCE", "SUMMER_BALANCE", "ANNUAL_BALANCE"]
    ]
    mass_balance = get_cumsum(mass_balance)
    return mass_balance.fillna(0)


def load_change(change: pd.DataFrame) -> pd.DataFrame:
    """ Load and preprocess change data.

    Parameters
    ----------
    change : pd.DataFrame
        Change data.

    Returns
    -------
    pd.DataFrame
        Preprocessed change data.
    """

    # change = change.drop_duplicates(subset=["WGMS_ID"])
    change = change.dropna(subset=["WGMS_ID"])
    change = change.loc[change["YEAR"] >= 2000, ["WGMS_ID", "YEAR", "AREA_CHANGE", "THICKNESS_CHG", "VOLUME_CHANGE"]]
    change = get_cumsum(change)
    return change.fillna(0)


def load_state(state: pd.DataFrame) -> pd.DataFrame:
    """ Load and preprocess state data.

    Parameters
    ----------
    state : pd.DataFrame
        State data.

    Returns
    -------
    pd.DataFrame
        Preprocessed state data.
    """

    # state = state.drop_duplicates(subset=["WGMS_ID"])
    state = state.dropna(subset=["WGMS_ID"])
    state = state.sort_values(by=["WGMS_ID", "YEAR"], ascending=False)
    state = state.loc[
        state["YEAR"] >= 2000,
        ["WGMS_ID", "YEAR", "AREA", "LENGTH", "HIGHEST_ELEVATION", "MEDIAN_ELEVATION", "LOWEST_ELEVATION"]
    ].sort_values(["YEAR", "WGMS_ID"])
    return state


def merge_data(
        loaded_glacier: pd.DataFrame,
        loaded_change: pd.DataFrame,
        loaded_state: pd.DataFrame,
        loaded_mass_balance: pd.DataFrame) -> pd.DataFrame:
    """ Merge data.

    Parameters
    ----------
    loaded_glacier : pd.DataFrame
        Preprocessed glaciers.
    loaded_change : pd.DataFrame
        Preprocessed change data.
    loaded_state : pd.DataFrame
        Preprocessed state data.
    loaded_mass_balance : pd.DataFrame
        Preprocessed mass balance data.

    Returns
    -------
    pd.DataFrame
        Merged data.
    """

    merged_data = loaded_glacier.merge(loaded_change, on="WGMS_ID", how="right")
    merged_data = merged_data.merge(loaded_state, on=["WGMS_ID", "YEAR"], how="outer")
    merged_data = merged_data.merge(loaded_mass_balance, on=["WGMS_ID", "YEAR"], how="outer")
    # log.warning(merged_data.isna().sum())
    return merged_data


def to_timeseries(merged_data: pd.DataFrame, parameters: Dict) -> Tuple[np.array, pd.DataFrame]:
    """ Convert merged data to timeseries for each WGMS_ID.

    Parameters
    ----------
    merged_data : pd.DataFrame
        Merged data.

    Returns
    -------
    np.array
        Array of Timeseries data.
    pd.DataFrame
        Glaciers with WGMS_ID and LATITUDE, LONGITUDE.
    """

    # merged_data = merged_data.iloc[:500000, :]

    features = parameters["features"]
    merged_data.dropna(subset=features, how='all', axis=0, inplace=True)  # how='all',
    merged_data = pd.concat(g for _, g in merged_data.groupby("WGMS_ID") if len(g) > 1)
    merged_data = pd.concat(g for _, g in merged_data.groupby("YEAR") if len(g) > 1)
    # merged_data = pd.concat(g for _, g in merged_data.groupby("WGMS_ID") if len(g) > 1)
    # merged_data = pd.concat(g for _, g in merged_data.groupby("WGMS_ID") if len(g) > 1)
    # merged_data = pd.concat(g for _, g in merged_data.groupby("WGMS_ID") if len(g) > 1)

    x = merged_data.WGMS_ID.factorize()[0]
    y = merged_data.groupby(x).cumcount().values
    vals = merged_data.loc[:, features].values
    out_shp = (x.max() + 1, y.max() + 1, vals.shape[1])
    out = np.full(out_shp, np.nan)
    out[x, y] = vals

    timeseries = to_time_series_dataset(out)

    reference_data = merged_data.set_index(["WGMS_ID", "YEAR"])

    print(timeseries.shape)
    print(reference_data.shape)

    return timeseries, reference_data
