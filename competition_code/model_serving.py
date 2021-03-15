import argparse
from utils import load_config
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime
import pandas as pd


def final_metric(low_corr, high_corr):
    # Metric as defined on the page
    # https://signate.jp/competitions/423#evaluation
    return (low_corr - 1) ** 2 + (high_corr - 1) ** 2


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help="Configuration ID for training"
    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Model Serving!!!!!")
    config = load_config(config_id)
    y_test_high = pd.read_csv("data/processed/y_test_high.csv")
    y_test_low = pd.read_csv("data/processed/y_test_low.csv")
    y_test_high.fillna(method="ffill", inplace=True)
    y_test_low.fillna(method="ffill", inplace=True)

    if config["lgb_model"]:
        print("Making predictions with LighGBMs!!!!!")
        test_tree = pd.read_csv("data/processed/test_trees.csv").fillna(0)
        print(test_tree.shape, y_test_high.shape)
        model_low = lgb.Booster(model_file="models/lgb_label_low_20.txt")
        model_high = lgb.Booster(model_file="models/lgb_label_high_20.txt")

        t1 = datetime.now()
        high_preds = model_high.predict(test_tree)
        low_preds = model_low.predict(test_tree)
        t2 = datetime.now()
        print(f"Inference Time is {(t2 - t1).seconds} seconds.")
        high_df = pd.concat([y_test_high, pd.DataFrame(high_preds)], axis=1)
        low_df = pd.concat([y_test_low, pd.DataFrame(low_preds)], axis=1)
        # Evaluate LightGBM
        spearman_high = spearmanr(y_test_high["label_high_20"], high_preds)[0]
        spearman_low = spearmanr(y_test_low["label_low_20"], low_preds)[0]
        print(spearman_high, spearman_low)
        final_metric = final_metric(spearman_low, spearman_high)

        print(
            "LightGBM Regressor: Final Leaderboard Score- Public - Test Set is "
            "same as public leaderboard."
        )
        print(final_metric)
