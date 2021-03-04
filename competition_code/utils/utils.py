import yaml
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


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
