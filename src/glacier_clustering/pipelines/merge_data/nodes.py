import pandas as pd
from logging import getLogger

log = getLogger(__name__)

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

    mass_balance = mass_balance.drop_duplicates(subset=["WGMS_ID"])
    mass_balance = mass_balance.dropna(subset=["WGMS_ID"])
    mass_balance = mass_balance[["WGMS_ID", "YEAR", "WINTER_BALANCE", "SUMMER_BALANCE", "ANNUAL_BALANCE"]]
    mass_balance = mass_balance.sort_values(by=["WGMS_ID", "YEAR"], ascending=False).drop_duplicates(subset=["WGMS_ID"])
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

    change = change.drop_duplicates(subset=["WGMS_ID"])
    change = change.dropna(subset=["WGMS_ID"])
    change = change[["WGMS_ID", "YEAR", "AREA_CHANGE", "THICKNESS_CHG", "VOLUME_CHANGE"]]
    change = change.sort_values(by=["WGMS_ID", "YEAR"], ascending=False).drop_duplicates(subset=["WGMS_ID"])
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

    state = state.drop_duplicates(subset=["WGMS_ID"])
    state = state.dropna(subset=["WGMS_ID"])
    state = state[["WGMS_ID", "YEAR", "AREA", "LENGTH", "HIGHEST_ELEVATION", "MEDIAN_ELEVATION", "LOWEST_ELEVATION"]]
    state = state.sort_values(by=["WGMS_ID", "YEAR"], ascending=False).drop_duplicates(subset=["WGMS_ID"])
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

    merged_data = loaded_glacier.merge(loaded_change, on="WGMS_ID", how="left")
    merged_data = merged_data.merge(loaded_state, on="WGMS_ID", how="left")
    merged_data = merged_data.merge(loaded_mass_balance, on="WGMS_ID", how="left")
    log.warning(merged_data.isna().sum())
    return merged_data.fillna(0)
