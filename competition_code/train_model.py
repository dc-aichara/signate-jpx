import argparse
from utils import load_config, lgb_spearmanr, final_metric
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from scipy.stats import spearmanr
import numpy as np


def train_single_lgb(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    param: dict,
    save_path: str,
) -> lgb.Booster:
    """Trains LightGBM model

    Args:
        X_train (pd.DataFrame): X_train
        X_valid (pd.DataFrame): X_valid
        y_train (np.ndarray): y_train
        y_valid (np.ndarray): y_valid
        param (dict): model hyperparameters
        save_path (str): path to save model

    Returns:
        lgb.Booster: Trained LightGBM model
    """
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    valid_sets = [lgb_train]
    if isinstance(X_valid, pd.DataFrame):
        lgb_valid = lgb.Dataset(data=X_valid, label=y_valid)
        valid_sets = [lgb_train, lgb_valid]
    model = lgb.train(
        param, lgb_train, valid_sets=valid_sets, verbose_eval=10, feval=lgb_spearmanr
    )
    featimp = model.feature_importance(importance_type="gain")
    feat_importance = pd.DataFrame(
        {"features": X_train.columns, "importance": featimp}
    ).sort_values(by="importance", ascending=False)
    if save_path:
        model.save_model(f"{save_path}.txt", num_iteration=model.best_iteration)
        feat_importance.to_csv(f"{save_path}.csv", index=False)

    else:
        model.save_model(f"models/single_lgbm.txt", num_iteration=model.best_iteration)
        feat_importance.to_csv("models/feat_imp.csv", index=False)

    return model


def cross_validation_lgbm(data: pd.DataFrame, param: dict, n_splits: int = 5) -> list:
    """Performs cross-validation for internal evaluation

    Args:
        data (pd.DataFrame): Full dataset
        param (dict): LightGBM Hyperparameters
        n_splits ([int], optional): Number of cv splits. Defaults to 5:int.

    Returns:
        list: A list of all CV scores for every split
    """
    lgbm_cv_scores = []

    folds = TimeSeriesSplit(n_splits=n_splits)

    for i, (train_index, test_index) in enumerate(folds.split(data)):
        print(f"FOLD - {i}")
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        valid_max = train_index.max()
        valid_min = int(len(train) * (80 / 100))

        train_cv = train.loc[0:valid_min]
        valid_cv = train.loc[valid_min:valid_max]

        train_high_cv = y_train_high.loc[0:valid_min]
        valid_high_cv = y_train_high.loc[valid_min:valid_max]
        test_high_cv = y_train_high.iloc[test_index]

        train_low_cv = y_train_low.loc[0:valid_min]
        valid_low_cv = y_train_low.loc[valid_min:valid_max]
        test_low_cv = y_train_low.iloc[test_index]

        lgb_train = lgb.Dataset(data=train_cv, label=train_high_cv)
        lgb_valid = lgb.Dataset(data=valid_cv, label=valid_high_cv)

        model_high = lgb.train(
            param,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            verbose_eval=10,
            feval=lgb_spearmanr,
        )

        high_preds_cv = model_high.predict(test)

        lgb_train = lgb.Dataset(data=train_cv, label=train_low_cv)
        lgb_valid = lgb.Dataset(data=valid_cv, label=valid_low_cv)

        model_low = lgb.train(
            param,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            verbose_eval=10,
            feval=lgb_spearmanr,
        )

        low_preds_cv = model_low.predict(test)

        # Evaluate Ridge
        spearman_high = spearmanr(test_high_cv, high_preds_cv)[0]
        spearman_low = spearmanr(test_low_cv, low_preds_cv)[0]

        metric_cv = final_metric(spearman_low, spearman_high)

        lgbm_cv_scores.append(metric_cv)

    return lgbm_cv_scores


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument("--config_id", type=str, help="Configuration ID for training")

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Train Models!!!")
    config = load_config(config_id)

    y_train_high = pd.read_csv("data/processed/y_train_high.csv")
    y_train_low = pd.read_csv("data/processed/y_train_low.csv")

    print(y_train_high.isnull().sum())
    # TEMPORARY FILL NAS with 0 until we figure out why they are null -
    # only around 30 examples
    y_train_high.fillna(method="ffill", inplace=True)
    y_train_low.fillna(method="ffill", inplace=True)

    if config.get("test_model") == "public":
        test_tree = pd.read_csv("data/processed/test_trees.csv").fillna(0)
        y_test_high = pd.read_csv("data/processed/y_test_high.csv")
        y_test_low = pd.read_csv("data/processed/y_test_low.csv")
        y_test_high.fillna(method="ffill", inplace=True)
        y_test_low.fillna(method="ffill", inplace=True)
        original_data = pd.read_csv("data/interim/test_data.csv")

    if config["lgb_model"]:
        train_tree = pd.read_csv("data/processed/train_trees.csv").fillna(0)
        print(train_tree.shape)
        params = config["lgb_params"]
        seed = config["seed"]
        valid_with_test = config.get("use_test_as_validation")

        if config["cross_validation"] is True:
            # Cross Validation
            lgbm_cv_scores = cross_validation_lgbm(train_tree, params, n_splits=5)
            print("LGBM CV scores")
            print(lgbm_cv_scores)
            print(np.mean(lgbm_cv_scores))

        print("Training  LightGBM!!!!!")
        # Use test_data data as validation data
        if valid_with_test and config.get("test_model") == "public":
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
        elif config.get("train_with_all_data"):
            (X_train_low, X_valid_low, y_train_low, y_valid_low) = (
                train_tree,
                None,
                y_train_low,
                None,
            )
            (X_train_high, X_valid_high, y_train_high, y_valid_high) = (
                train_tree,
                None,
                y_train_high,
                None,
            )
        else:
            (X_train_low, X_valid_low, y_train_low, y_valid_low) = train_test_split(
                train_tree, y_train_low, test_size=0.2, shuffle=False
            )
            (X_train_high, X_valid_high, y_train_high, y_valid_high) = train_test_split(
                train_tree, y_train_high, test_size=0.2, shuffle=False
            )

        print(X_train_low.shape, y_train_low.shape)
        model_low = train_single_lgb(
            X_train_low,
            X_valid_low,
            y_train_low,
            y_valid_low,
            params,
            "models/lgb_label_low_20",
        )
        params["num_iterations"] = 100
        model_high = train_single_lgb(
            X_train_high,
            X_valid_high,
            y_train_high,
            y_valid_high,
            params,
            "models/lgb_label_high_20",
        )

        if config.get("test_model") == "public":
            high_preds = model_high.predict(test_tree)
            low_preds = model_low.predict(test_tree)

            high_df = pd.concat([y_test_high, pd.DataFrame(high_preds)], axis=1)
            low_df = pd.concat([y_test_low, pd.DataFrame(low_preds)], axis=1)

            # Evaluate LightGBM
            spearman_high = spearmanr(y_test_high, high_preds)[0]
            spearman_low = spearmanr(y_test_low, low_preds)[0]
            print(spearman_low, spearman_high)
            final_metric_tree = final_metric(spearman_low, spearman_high)

            original_data["lgbm_high"] = high_preds
            original_data["lgbm_low"] = low_preds

            print(
                "LightGBM Regressor: Final Leaderboard Score- Public - Test "
                "Set is same as public leaderboard."
            )
            print(final_metric_tree)
            # Tree Based Model

            high_df.to_csv("data/submissions/high_df.csv")
            low_df.to_csv("data/submissions/low_df.csv")

            original_data.to_csv("data/error_analysis/preds.csv", index=False)
