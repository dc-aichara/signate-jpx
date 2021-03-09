import argparse
from utils import load_config, lgb_spearmanr
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.stats import spearmanr


def final_metric(low_corr, high_corr):
    # Metric as defined on the page
    # https://signate.jp/competitions/423#evaluation
    return (low_corr - 1) ** 2 + (high_corr - 1) ** 2


def train_single_lgb(X_train, X_valid, y_train, y_valid, param, save_path):
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_valid = lgb.Dataset(data=X_valid, label=y_valid)
    model = lgb.train(
        param,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=10,
        feval=lgb_spearmanr,
        # categorical_feature=cat_feats
    )
    featimp = model.feature_importance(importance_type="gain")
    feat_importance = pd.DataFrame(
        {"features": X_train.columns, "importance": featimp}
    ).sort_values(by="importance", ascending=False)
    if save_path:
        model.save_model(f"{save_path}.txt", num_iteration=model.best_iteration)
        feat_importance.to_csv(f"{save_path}.csv", index=False)

    else:
        model.save_model(
            f"models/single_lgbm.txt", num_iteration=model.best_iteration
        )
        feat_importance.to_csv("models/feat_imp.csv", index=False)

    return model


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help="Configuration ID for training"
    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Train Models!!!")
    config = load_config(config_id)

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
    print(y_train_high.shape, y_train_low.shape, y_test_high.shape, y_test_low.shape)
    # Ridge Regression
    if config["ridge_regression"]:
        train_linear = pd.read_csv("data/processed/train_linear.csv").fillna(0)
        test_linear = pd.read_csv("data/processed/test_linear.csv").fillna(0)
        alpha = config["alpha"]
        print(f"Training  Ridge Regressor with Alpha = {alpha}!!!!!")
        Ridge_high = linear_model.Ridge(alpha=alpha)
        Ridge_low = linear_model.Ridge(alpha=alpha)

        Ridge_high.fit(train_linear, y_train_high)
        Ridge_low.fit(train_linear, y_train_low)

        high_preds = Ridge_high.predict(test_linear)
        low_preds = Ridge_low.predict(test_linear)

        high_df = pd.concat([y_test_high, pd.DataFrame(high_preds)], axis=1)
        low_df = pd.concat([y_test_low, pd.DataFrame(low_preds)], axis=1)

        # Evaluate Ridge
        spearman_high = spearmanr(y_test_high, high_preds)[0]
        spearman_low = spearmanr(y_test_low, low_preds)[0]

        final_metric_linear = final_metric(spearman_low, spearman_high)

        print(
            "Ridge Regressor: Final Leaderboard Score- Public - Test Set is "
            "same as public leaderboard."
        )
        print(final_metric_linear)

    if config["lgb_model"]:
        train_tree = pd.read_csv("data/processed/train_trees.csv").fillna(0)
        test_tree = pd.read_csv("data/processed/test_trees.csv").fillna(0)
        params = config["lgb_params"]
        seed = config["seed"]
        valid_with_test = config.get("use_test_as_validation")
        print("Training  LightGBM!!!!!")
        # Use test_data data as validation data
        if valid_with_test:
            print("Using Test data a validation data!!!")
            X_train_low, X_valid_low, y_train_low, y_valid_low = (
                train_tree,
                test_tree,
                y_train_low,
                y_test_low,
            )
            X_train_high, X_valid_high, y_train_high, y_valid_high = (
                train_tree,
                test_tree,
                y_train_high,
                y_test_high,
            )
        else:
            (
                X_train_low,
                X_valid_low,
                y_train_low,
                y_valid_low,
            ) = train_test_split(
                train_tree, y_train_low, test_size=0.2, random_state=seed
            )
            (
                X_train_high,
                X_valid_high,
                y_train_high,
                y_valid_high,
            ) = train_test_split(
                train_tree, y_train_high, test_size=0.2, random_state=seed
            )

        print(X_train_low.shape, X_valid_low.shape, y_train_low.shape,y_valid_low.shape)
        model_low = train_single_lgb(
            X_train_low,
            X_valid_low,
            y_train_low,
            y_valid_low,
            params,
            "models/low_lgb",
        )
        model_high = train_single_lgb(
            X_train_high,
            X_valid_high,
            y_train_high,
            y_valid_high,
            params,
            "models/high_lgb",
        )

        high_preds = model_high.predict(test_tree)
        low_preds = model_low.predict(test_tree)

        high_df = pd.concat([y_test_high, pd.DataFrame(high_preds)], axis=1)
        low_df = pd.concat([y_test_low, pd.DataFrame(low_preds)], axis=1)

        # Evaluate LightGBM
        spearman_high = spearmanr(y_test_high, high_preds)[0]
        spearman_low = spearmanr(y_test_low, low_preds)[0]
        print(spearman_low, spearman_high)
        final_metric_tree = final_metric(spearman_low, spearman_high)

        print(
            "LightGBM Regressor: Final Leaderboard Score- Public - Test Set is "
            "same as public leaderboard."
        )
        print(final_metric_tree)

    # Tree Based Model - For Later

    high_df.to_csv("data/submissions/high_df.csv")
    low_df.to_csv("data/submissions/low_df.csv")
