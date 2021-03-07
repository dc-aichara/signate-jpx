import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from PriceIndices import Indices


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def load_config(config_id: str) -> dict:
    with open("config.yml", "r") as f:
        doc = yaml.load(f, yaml.Loader)

    config = doc[config_id]
    return config


def time_series_CV(data: pd.DataFrame, n_splits: int = 5):

    folds = TimeSeriesSplit(n_splits=n_splits)

    for i, (train_index, test_index) in enumerate(folds.split(data)):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        print(f"FOLD- {i}")
        print(f"Train Min-{train.index.min()}, Train Max- {train.index.max()}")
        print(f"Test Min-{test.index.min()}, Train Max- {test.index.max()}")

        yield train, test


def calculate_price_indices(
    data: pd.DataFrame, date_col: str = "date", price_col: str = "price"
) -> pd.DataFrame:
    """
    Calculate share price technical indicators
    Args:
        data (pd.DataFrame): Pandas Dataframe
        date_col (str): Date column in DataFrame
        price_col (str): Price column in DataFrame

    Returns:
        pd.DataFrame: A Pandas DataFrame with price indicators as columns.

    """
    df = data[[date_col, price_col]]
    indices = Indices(df, date_col=date_col, price_col=price_col)
    df_bi = indices.get_vola_index()
    df_bi.drop(price_col, axis=1, inplace=True)
    del df_bi
    df_rsi = indices.get_rsi()
    df_rsi.drop([price_col, "RS_Smooth", "RSI_1"], axis=1, inplace=True)
    df_sma = indices.get_simple_moving_average()
    df_sma.drop([price_col], axis=1, inplace=True)
    df_bb = indices.get_bollinger_bands()
    df_bb.drop([price_col], axis=1, inplace=True)
    df_ema = indices.get_exponential_moving_average([20, 50])
    df_ema.drop([price_col], axis=1, inplace=True)
    df_macd = indices.get_moving_average_convergence_divergence()
    df_macd.drop([price_col], axis=1, inplace=True)

    df = pd.merge(df, df_macd, on=date_col, how="left")
    df = pd.merge(df, df_rsi, on=date_col, how="left")
    df = pd.merge(df, df_bi, on=date_col, how="left")
    df = pd.merge(df, df_bb, on=date_col, how="left")
    df = pd.merge(df, df_ema, on=date_col, how="left")
    df = pd.merge(df, df_sma, on=date_col, how="left")
    del df_rsi, df_macd, df_sma, df_bb, df_bi, df_ema
    df.rename(columns={"RSI_2": "RSI"}, inplace=True)
    df.fillna(0)
    for col in df.columns[:-8]:
        df[col] = np.round(df[col], 2)
    df.sort_values(date_col, ascending=False, inplace=True)

    return df


def date_feats(
    df: pd.DataFrame,
    date_col: str = "date",
    month_col: str = "month",
    dom_col: str = "dayofmonth",
    dow_col: str = "dayofweek",
) -> pd.DataFrame:
    """
    Create Month, DayOfMonth, and DayOfWeek features given DataFrame with date
    column.
    Args:
        df (pd.DataFrame): Pandas DataFrame
        date_col (str): Date column in DataFrame
        month_col (str): Month column
        dom_col (str): DayOfMonth column
        dow_col (str): DayOfWeek column

    Returns:
        pd.DataFrame: A DataFrame with three new columns.

    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data[month_col] = data[date_col].dt.month
    data[dom_col] = data[date_col].dt.day
    data[dow_col] = data[date_col].dt.dayofweek

    return data
