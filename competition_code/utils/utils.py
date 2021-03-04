import yaml
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from PriceIndices import Indices


def load_config(config_id):
    with open("config.yml", "r") as f:
        doc = yaml.load(f, yaml.Loader)

    config = doc[config_id]
    return config


def time_series_CV(data: pd.DataFrame, n_splits: int = 5):

   folds = TimeSeriesSplit(n_splits=n_splits)

   for i, (train_index, test_index) in enumerate(folds.split(data)):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        print(f'FOLD- {i}')
        print(f'Train Min-{train.index.min()}, Train Max- {train.index.max()}')
        print(f'Test Min-{test.index.min()}, Train Max- {test.index.max()}')

        yield train, test


def calculate_price_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate share price technical indicators
    Args:
        df (pd.DataFrame): Pandas Dataframe

    Returns:
        pd.DataFrame

    """
    indices = Indices(df, date_col="date", price_col="price")
    df_bi = indices.get_vola_index()
    df_bi.drop("price", axis=1, inplace=True)
    df_rsi = indices.get_rsi()
    df_rsi.drop(["price", "RS_Smooth", "RSI_1"], axis=1, inplace=True)
    df_sma = indices.get_simple_moving_average()
    df_sma.drop(["price"], axis=1, inplace=True)
    df_bb = indices.get_bollinger_bands()
    df_bb.drop(["price"], axis=1, inplace=True)
    df_ema = indices.get_exponential_moving_average([20, 50])
    df_ema.drop(["price"], axis=1, inplace=True)
    df_macd = indices.get_moving_average_convergence_divergence()
    df_macd.drop(["price"], axis=1, inplace=True)

    df = pd.merge(df, df_macd, on="date", how="left")
    df = pd.merge(df, df_rsi, on="date", how="left")
    df = pd.merge(df, df_bi, on="date", how="left")
    df = pd.merge(df, df_bb, on="date", how="left")
    df = pd.merge(df, df_ema, on="date", how="left")
    df = pd.merge(df, df_sma, on="date", how="left")
    del df_rsi, df_macd, df_sma, df_bb, df_bi
    df.rename(columns={"RSI_2": "RSI"}, inplace=True)
    df.fillna(0)
    for col in df.columns[:-8]:
        df[col] = np.round(df[col], 2)
    df.sort_values("date", ascending=False, inplace=True)

