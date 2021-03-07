import argparse
from utils import load_config
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from scipy.stats import spearmanr


def final_metric(spearman_low, spearman_high):
    # Metric as defined on the page https://signate.jp/competitions/423#evaluation
    return (spearman_low - 1) ** 2 + (spearman_high - 1) ** 2


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help="Configuration ID for training"
    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Train Models!!!")
    config = load_config(config_id)

    train_linear = pd.read_csv("data/processed/train_linear.csv").fillna(0)
    test_linear = pd.read_csv("data/processed/test_linear.csv").fillna(0)

    y_train_high = pd.read_csv("data/processed/y_train_high.csv")
    y_train_low = pd.read_csv("data/processed/y_train_low.csv")
    y_test_high = pd.read_csv("data/processed/y_test_high.csv")
    y_test_low = pd.read_csv("data/processed/y_test_low.csv")

    print(y_train_high.isnull().sum())

    #### TEMPORARY FILL NAS with 0 until we figure out why they are null - only around 30 examples
    y_train_high.fillna(method="ffill", inplace=True)
    y_train_low.fillna(method="ffill", inplace=True)

    y_test_high.fillna(method="ffill", inplace=True)
    y_test_low.fillna(method="ffill", inplace=True)

    # Ridge Regression
    Ridge_high = linear_model.Ridge(alpha=0.001)
    Ridge_low = linear_model.Ridge(alpha=0.001)

    Ridge_high.fit(train_linear, y_train_high)
    Ridge_low.fit(train_linear, y_train_low)

    high_preds = Ridge_high.predict(test_linear)
    low_preds = Ridge_low.predict(test_linear)

    high_df = pd.concat([y_test_high, pd.DataFrame(high_preds)], axis=1)
    low_df = pd.concat([y_test_low, pd.DataFrame(low_preds)], axis=1)

    # Evaluate Ridge
    spearman_high = spearmanr(y_test_high, high_preds)[0]
    spearman_low = spearmanr(y_test_low, low_preds)[0]

    final_metric = final_metric(spearman_low, spearman_high)

    print(
        "Final Leaderboard Score- Public - Test Set is same as public "
        "leaderboard."
    )
    print(final_metric)

    # Tree Based Model - For Later

    high_df.to_csv("data/submissions/high_df.csv")
    low_df.to_csv("data/submissions/low_df.csv")
