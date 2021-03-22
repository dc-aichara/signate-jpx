import argparse
from utils import load_config, lgb_spearmanr, time_series_CV
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from scipy.stats import spearmanr
import numpy as np

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from pytorch_tabnet.metrics import Metric
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle


def final_metric(low_corr, high_corr):
    # Metric as defined on the page
    # https://signate.jp/competitions/423#evaluation
    return (low_corr - 1) ** 2 + (high_corr - 1) ** 2


def train_single_lgb(X_train, X_valid, y_train, y_valid, param, save_path):
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    valid_sets = [lgb_train]
    if isinstance(X_valid, pd.DataFrame):
        lgb_valid = lgb.Dataset(data=X_valid, label=y_valid)
        valid_sets = [lgb_train, lgb_valid]
    model = lgb.train(
        param,
        lgb_train,
        valid_sets=valid_sets,
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


def cross_validation_ridge(data, n_splits=5):
    ridge_cv_scores = []

    folds = TimeSeriesSplit(n_splits=n_splits)

    for i, (train_index, test_index) in enumerate(folds.split(data)):
        train_cv = data.iloc[train_index]
        test_cv = data.iloc[test_index]

        train_high_cv = y_train_high.iloc[train_index]
        test_high_cv = y_train_high.iloc[test_index]

        train_low_cv = y_train_low.iloc[train_index]
        test_low_cv = y_train_low.iloc[test_index]

        model_high = linear_model.Ridge(alpha=alpha)
        model_high.fit(train_cv, train_high_cv)

        model_low = linear_model.Ridge(alpha=alpha)
        model_low.fit(train_cv, train_low_cv)

        high_preds_cv = model_high.predict(test_cv)
        low_preds_cv = model_low.predict(test_cv)

        # Evaluate Ridge
        spearman_high = spearmanr(test_high_cv, high_preds_cv)[0]
        spearman_low = spearmanr(test_low_cv, low_preds_cv)[0]

        metric_cv = final_metric(spearman_low, spearman_high)

        ridge_cv_scores.append(metric_cv)

    return ridge_cv_scores


def cross_validation_lgbm(data, param, n_splits=5):
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

    # TEMPORARY FILL NAS with 0 until we figure out why they are null -
    # only around 30 examples
    y_train_high.fillna(method="ffill", inplace=True)
    y_train_low.fillna(method="ffill", inplace=True)

    y_test_high.fillna(method="ffill", inplace=True)
    y_test_low.fillna(method="ffill", inplace=True)
    print(
        y_train_high.shape,
        y_train_low.shape,
        y_test_high.shape,
        y_test_low.shape,
    )
    original_data = pd.read_csv("data/interim/test_data.csv")
    # Ridge Regression
    if config["ridge_regression"]:
        train_linear = pd.read_csv("data/processed/train_linear.csv").fillna(0)
        test_linear = pd.read_csv("data/processed/test_linear.csv").fillna(0)
        alpha = config["alpha"]
        print(train_linear.shape, test_linear.shape)
        if config["cross_validation"] is True:
            print("Begin Ridge- Cross Validation")

            ridge_cv_scores = cross_validation_ridge(train_linear, n_splits=5)

            print("Ridge Cross Validation Results")
            print(ridge_cv_scores)
            print(np.mean(ridge_cv_scores))

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

        original_data["rr_high"] = high_preds
        original_data["rr_low"] = low_preds

        final_metric_linear = final_metric(spearman_low, spearman_high)

        print(
            "Ridge Regressor: Final Leaderboard Score- Public - Test Set is "
            "same as public leaderboard."
        )
        print(spearman_low, spearman_high)
        print(final_metric_linear)

    if config['tab_net']:

        valid_with_test = config.get("use_test_as_validation")  

        train_tree = pd.read_csv("data/processed/train_trees.csv").fillna(0)
        test_tree = pd.read_csv("data/processed/test_trees.csv").fillna(0)

        if valid_with_test:
            print("Using Test data a validation data!!!")
            X_train_tn_low, X_valid_tn_low, y_train_tn_low, y_valid_tn_low = (
                train_tree,
                test_tree,
                y_train_low,
                y_test_low,
            )
            X_train_tn_high, X_valid_tn_high, y_train_tn_high, y_valid_tn_high = (
                train_tree,
                test_tree,
                y_train_high,
                y_test_high,
            )
        elif config.get("train_with_all_data"):
            X_train_tn_low, X_valid_tn_low, y_train_tn_low, y_valid_tn_low = train_test_split(train_tree, y_train_low, test_size=0.2, shuffle=False)
            X_train_tn_high, X_valid_tn_high, y_train_tn_high, y_valid_tn_high = train_test_split(train_tree, y_train_high, test_size=0.2, shuffle=False)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            print("working on gpu")

        class spearman(Metric):
            def __init__(self):
                self._name = "spearman"
                self._maximize = True

            def __call__(self, y_true, y_pred):
                corr = spearmanr(y_true, y_pred)[0]
                return corr

        # Model Params 
        tabnet_params = dict(n_d = 32,
                             n_a = 32,
                             n_steps = 5,
                             gamma = 1.3,
                             lambda_sparse = 0,
                             optimizer_fn = optim.Adam,
                             optimizer_params = dict(lr = .1, weight_decay = 1e-5),
                             mask_type = "entmax",
                             scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
                             scheduler_fn = ReduceLROnPlateau,
                             seed = 42,
                             verbose = 1)

        tabnet_model_low = TabNetRegressor(**tabnet_params, device_name='auto')

        tabnet_model_low.fit(X_train_tn_low.to_numpy(), 
                             y_train_tn_low.to_numpy(), 
                             eval_set=[(X_valid_tn_low.to_numpy(), y_valid_tn_low.to_numpy())],
                             eval_metric=[spearman],
                             patience=20,
                             batch_size=int(32768*2), 
                             virtual_batch_size=int(4096*2),
                             num_workers=4,
                             max_epochs=500,
                             drop_last=False)
        tabnet_low_preds = tabnet_model_low.predict(test_tree.to_numpy())
        
        tabnet_model_high = TabNetRegressor(**tabnet_params, device_name='auto')
        tabnet_model_high.fit(X_train_tn_high.to_numpy(),
                              y_train_tn_high.to_numpy(), 
                              eval_set=[(X_valid_tn_high.to_numpy(), y_valid_tn_high.to_numpy())],
                              eval_metric=[spearman],
                              patience=20,
                              batch_size=int(32768*2), 
                              virtual_batch_size=int(4096*2),
                              num_workers=4,
                              max_epochs=500,
                              drop_last=False)
        tabnet_high_preds = tabnet_model_high.predict(test_tree.to_numpy())

        # Evaluate LightGBM
        spearman_tn_high = spearmanr(y_test_high, tabnet_high_preds)[0]
        spearman_tn_low = spearmanr(y_test_low, tabnet_low_preds)[0]
        print(spearman_tn_low, spearman_tn_high)
        final_metric_tabnet = final_metric(spearman_tn_low, spearman_tn_high)

        original_data['tabnet_high'] = tabnet_high_preds
        original_data['tabnet_low'] = tabnet_low_preds

        print('Tab Net')
        print(final_metric_tabnet)

        tabnet_model_low.save_model("models/tabnet_label_low_20")
        tabnet_model_high.save_model("models/tabnet_label_high_20")

    if config["lgb_model"]:
        train_tree = pd.read_csv("data/processed/train_trees.csv").fillna(0)
        test_tree = pd.read_csv("data/processed/test_trees.csv").fillna(0)

        
        print(train_tree.shape, test_tree.shape)
        params = config["lgb_params"]
        seed = config["seed"]
        valid_with_test = config.get("use_test_as_validation")

        if config["cross_validation"] is True:
            # Cross Validation
            lgbm_cv_scores = cross_validation_lgbm(
                train_tree, params, n_splits=5
            )
            print("LGBM CV scores")
            print(lgbm_cv_scores)
            print(np.mean(lgbm_cv_scores))

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
        elif config.get("train_with_all_data"):
            (X_train_low, X_valid_low, y_train_low, y_valid_low,) = (
                train_tree,
                None,
                y_train_low,
                None,
            )
            (X_train_high, X_valid_high, y_train_high, y_valid_high,) = (
                train_tree,
                None,
                y_train_high,
                None,
            )
        else:
            (
                X_train_low,
                X_valid_low,
                y_train_low,
                y_valid_low,
            ) = train_test_split(
                train_tree, y_train_low, test_size=0.2, shuffle=False
            )
            (
                X_train_high,
                X_valid_high,
                y_train_high,
                y_valid_high,
            ) = train_test_split(
                train_tree, y_train_high, test_size=0.2, shuffle=False
            )

        print(
            X_train_low.shape,
            y_train_low.shape,
        )
        model_low = train_single_lgb(
            X_train_low,
            X_valid_low,
            y_train_low,
            y_valid_low,
            params,
            "models/lgb_label_low_20",
        )
        # params["num_iterations"] = 200
        model_high = train_single_lgb(
            X_train_high,
            X_valid_high,
            y_train_high,
            y_valid_high,
            params,
            "models/lgb_label_high_20",
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

        original_data["lgbm_high"] = high_preds
        original_data["lgbm_low"] = low_preds

        print(
            "LightGBM Regressor: Final Leaderboard Score- Public - Test Set is "
            "same as public leaderboard."
        )
        print(final_metric_tree)

    # Tree Based Model - For Later

    high_df.to_csv("data/submissions/high_df.csv")
    low_df.to_csv("data/submissions/low_df.csv")

    original_data.to_csv("data/error_analysis/preds.csv", index=False)
