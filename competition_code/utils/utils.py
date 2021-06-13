import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from PriceIndices import Indices
import sklearn
from typing import Tuple, Iterator


def load_data(data_dir: str = "data/raw/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Given a path to the data will load all datasets into pandas dataframes. 

    Args:
        data_dir (str, optional): Path to the input datasets where stock_labels, stock_fin, stock_list, and stock_price are located. Defaults to "data/raw/".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: stock_price df, stock_fin df, stock_list df, stock_labels df
    """    

    
    print(f"Loading data from {data_dir}")
    stock_labels = pd.read_csv(f"{data_dir}stock_labels.csv.gz")
    stock_fin = pd.read_csv(f"{data_dir}stock_fin.csv.gz")
    stock_list = pd.read_csv(f"{data_dir}stock_list.csv.gz")
    stock_price = pd.read_csv(f"{data_dir}stock_price.csv.gz")

    return stock_price, stock_fin, stock_list, stock_labels


def format_dates(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Automatically formats all specified columns as date format "%Y-%m-%d"

    Args:
        df (pd.DataFrame): Original DataFrame of data
        columns (list): List of columns that should be dates 

    Returns:
        pd.DataFrame: Returns original Dataframe with cleaned up dates. 
    """    

    for column in columns:
        df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d")

    return df


def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Function for reducing memory usage by downcasting of types
       in dataframes. 

    Args:
        df (pd.DataFrame): DataFrame of Data
        verbose (bool, optional): Whether to print results. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe of data after type downcasting
    """    
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
    """Reads Configuration from yaml file. 

    Args:
        config_id (str): specific configuration ID to use in yaml file

    Returns:
        dict: Returns dictionary of configuration parameters
    """    
    with open("config.yml", "r") as f:
        doc = yaml.load(f, yaml.Loader)

    config = doc[config_id]
    return config

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
    df = data.copy()
    indices = Indices(df, date_col=date_col, price_col=price_col)
    df_rsi = indices.get_rsi()
    df_rsi.drop([price_col, "RS_Smooth", "RSI_1"], axis=1, inplace=True)
    df_bb = indices.get_bollinger_bands()
    df_bb.drop([price_col], axis=1, inplace=True)
    df_macd = indices.get_moving_average_convergence_divergence()
    df_macd.drop([price_col], axis=1, inplace=True)

    df = pd.merge(df, df_macd, on=[date_col, "Local Code"], how="left")
    df = pd.merge(df, df_rsi, on=[date_col, "Local Code"], how="left")
    df = pd.merge(df, df_bb, on=[date_col, "Local Code"], how="left")
    del df_bb, df_macd, df_rsi
    df.rename(columns={"RSI_2": "RSI"}, inplace=True)
    df.fillna(0)
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


def lgb_spearmanr(preds, dtrain_data):
    """
    Spearman's rank correlation coefficient metrics for LightGBM
    """
    labels = dtrain_data.get_label()
    corr = spearmanr(labels, preds)[0]
    return "lgb_corr", corr, True


def lgb_r2_score(preds, dtrain_data):
    """
    R^2 metrics for LightGBM
    """
    labels = dtrain_data.get_label()
    return "r2", r2_score(labels, preds), True


def final_metric(low_corr, high_corr):
    # Metric as defined on the page
    # https://signate.jp/competitions/423#evaluation
    return (low_corr - 1) ** 2 + (high_corr - 1) ** 2


def get_data_rules(config: dict):
    numerics = []
    dates = []
    categoricals = []
    drops = []

    for key, value in config.items():
        if value == "numeric":
            numerics.append(key)
        if value == "date":
            dates.append(key)
        if value == "categorical":
            categoricals.append(key)
        if value == "drop":
            drops.append(key)

    return numerics, dates, categoricals, drops


def auto_categorical(df, encoder, categoricals):
    for category in categoricals:
        df[category] = df[category].fillna("no_category").astype(str)

    transformed_df = pd.DataFrame(encoder.transform(df[categoricals]))

    # if ohe
    if isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder):
        print("OHE")
        transformed_df.columns = encoder.get_feature_names()

    if isinstance(encoder, sklearn.preprocessing._encoders.OrdinalEncoder):
        print("Ordinal Encoder")
        transformed_df.columns = categoricals

    return transformed_df


def auto_numeric(df, scaler, numerics):
    numerics_df = pd.DataFrame(scaler.transform(df[numerics]))
    numerics_df.columns = numerics

    return numerics_df


def auto_dates(train, dates):
    # DF with all the dates
    dates_df = pd.DataFrame()
    for date in dates:
        dates_df[f"{date}_month"] = pd.to_datetime(train[date]).dt.month
        dates_df[f"{date}_day"] = pd.to_datetime(train[date]).dt.day
        dates_df[f"{date}_dayofweek"] = pd.to_datetime(train[date]).dt.dayofweek

    return dates_df


def get_technical_features(
    df,
    date_col="base_date",
    price_col="EndOfDayQuote ExchangeOfficialClose",
    periods=[7, 14, 21],
    extra_feats=False,
):
    """
    Args:
        df (pd.DataFrame): DataFrame
        date_col (str): Date column in DataFrame
        price_col (str): Price column in DataFrame
        periods (list): List of periods to create technical features
        extra_feats (bool): If create extra features from Priceindices
    Returns:
        pd.DataFrame: Feature DataFrame
    """
    data = df[["Local Code", date_col, price_col]]
    data = data.sort_values(date_col)
    datas = []
    for code in data["Local Code"].unique():
        feats = data[data["Local Code"] == code]
        if extra_feats:
            feats = calculate_price_indices(
                feats, date_col=date_col, price_col=price_col
            )
        feats[f"EMA_{periods[1]}"] = (
            feats[price_col].ewm(span=periods[2], adjust=False).mean()
        )
        feats[f"return_{periods[0]}"] = feats[price_col].pct_change(periods[0])
        feats[f"return_{periods[1]}"] = feats[price_col].pct_change(periods[1])
        feats[f"return_{periods[2]}"] = feats[price_col].pct_change(periods[2])
        feats[f"volatility_{periods[0]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[0]).std()
        )
        feats[f"volatility_{periods[1]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[2]).std()
        )
        feats[f"volatility_{periods[2]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[2]).std()
        )
        feats[f"MA_gap_{periods[0]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[0]).mean()
        )
        feats[f"MA_gap_{periods[1]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[1]).mean()
        )
        feats[f"MA_gap_{periods[2]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[2]).mean()
        )
        feats = feats.fillna(0)
        feats = feats.drop([price_col], axis=1)
        datas.append(feats)
    feats = pd.concat(datas, ignore_index=True)
    del datas, data
    return feats
