# -*- coding: utf-8 -*-
import io
import pickle
import pandas as pd
import lightgbm as lgb
from utils import (
    format_dates,
    load_config,
    get_technical_features,
    auto_numeric,
    auto_categorical,
    auto_dates,
)
import yaml
import json


class ScoringService(object):
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    start_dt = "2020-01-01"

    with open("config.yml", "r") as f:
        doc = yaml.load(f, yaml.Loader)

    config = doc["baseline_model"]

    # metadata
    with open("../model/metadata.json", "r") as f:
        metadata = json.load(f)

    print(metadata)

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            # "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_for_predict(cls, dfs):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        print(cls.config)

        stock_list = dfs["stock_list"]
        stock_price = dfs["stock_price"]
        stock_labels = dfs["stock_labels"]

        # Fix all Date Formats
        stock_price = format_dates(stock_price, ["EndOfDayQuote Date"])
        stock_price.rename(
            columns={"EndOfDayQuote Date": "base_date"}, inplace=True
        )
        stock_labels = format_dates(
            stock_labels,
            ["base_date", "label_date_5", "label_date_10", "label_date_20"],
        )

        # Filter Dates
        stock_price = stock_price[stock_price["base_date"] >= cls.start_dt]
        stock_labels = stock_labels[stock_labels["base_date"] >= cls.start_dt]

        feats = pd.merge(
            stock_labels, stock_list, on=["Local Code"], how="left"
        )

        feats = feats[feats["prediction_target"] == True]

        feats = pd.merge(
            feats, stock_price, on=["base_date", "Local Code"], how="left"
        )

        feats.reset_index(inplace=True, drop=True)

        # Get simple Technical features
        tech_feat1 = get_technical_features(
            feats, periods=[10, 20, 30], extra_feats=True
        )

        feats = pd.merge(
            feats, tech_feat1, on=["base_date", "Local Code"], how="left"
        )
        del tech_feat1
        tech_feat2 = get_technical_features(feats, periods=[14, 28, 42])
        feats = pd.merge(
            feats, tech_feat2, on=["base_date", "Local Code"], how="left"
        )
        del tech_feat2
        feats["change_pct"] = (
                (feats["EndOfDayQuote High"] - feats["EndOfDayQuote Low"])
                * 100
                / (feats["EndOfDayQuote Close"])
        )
        feats["change"] = feats["EndOfDayQuote Open"] - feats[
            "EndOfDayQuote Close"]

        feats.reset_index(drop=True, inplace=True)

        # Numeric Encoding
        # Load from file
        with open("../model/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        numeric_feats = auto_numeric(feats, scaler, cls.metadata["numeric"])

        # Categorical Encoding
        with open("../model/ordenc.pkl", "rb") as file:
            ordenc = pickle.load(file)

        categorical_feats = auto_categorical(
            feats, ordenc, cls.metadata["categorical"]
        )

        # Dates Encoding
        date_feats = auto_dates(feats, cls.metadata["dates"])

        dates = feats["base_date"]
        local_codes = feats["Local Code"]

        # Reindex columns

        feats = pd.concat(
            [date_feats, categorical_feats, numeric_feats], axis=1
        )

        feats["base_date"] = dates
        feats["Local Code"] = local_codes

        return feats

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method
        Args:
            model_path (str): Path to the trained model directory.
        Returns:
            bool: The return value. True for success, False otherwise.
        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(f"{model_path}/lgb_{label}.txt")
            cls.models[label] = lgb.Booster(
                model_file=f"{model_path}/lgb_{label}.txt"
            )
        return True

    @classmethod
    def predict(cls, inputs, labels=None, codes=None):
        """Predict method
        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
        Returns:
            dict[pd.DataFrame]: Inference for the given input.
        """
        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        feats = cls.get_features_for_predict(cls.dfs)

        print(feats.loc[:, "base_date"])

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        df = pd.DataFrame()

        df["code"] = pd.to_datetime(feats["base_date"]).dt.strftime(
            "%Y-%m-%d-"
        ) + feats["Local Code"].astype(str)

        feats.drop(columns=["base_date", "Local Code"], inplace=True)
        print(df["code"].head())

        # 出力対象列を定義
        output_columns = ["code"]

        # 目的変数毎に予測
        for label in labels:
            df[label] = cls.models[label].predict(feats)

            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()
