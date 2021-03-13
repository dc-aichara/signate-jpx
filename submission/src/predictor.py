# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

class ScoringService(object):
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]
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
            #"stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs
    
    @classmethod
    def get_dataset(cls, inputs):
        """ Args:
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
        """ Args:
                dfs (dict[pd.DataFrame]): loaded data
            Returns:
                array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
        "Local Code"].values
        return cls.codes
        

    @classmethod
    def get_features_for_predict(cls, dfs, code):
        """ 
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み、欠損値の多いカラムを削除
        stock_fin = (dfs["stock_fin"].copy().drop(["Result_Dividend DividendPayableDate",],axis=1, ))
        # 特定の銘柄コードのデータに絞る
        fin_data = stock_fin[stock_fin["Local Code"] == code]
        # 日付列をpd.Timestamp型に変換してindexに設定
        fin_data["datetime"] = pd.to_datetime(fin_data["base_date"])
        fin_data.set_index("datetime", inplace=True)
        
        # 特徴量追加（詳しくはチュートリアル参考）
        # チュートリアルでは特徴量を付加したdataframeをfeatsと命名しています
        # ここでは単純にfin_dataをコピーしているため、各自で特徴量を付加したdataframeをお使いください
        feats = fin_data.copy()
        feats['code'] = code
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
            cls.models[label] = lgb.Booster(model_file=f"{model_path}/lgb_{label}.txt") 
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
        
        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)

        # 出力対象列を定義
        output_columns = ["code"]

        # 目的変数毎に予測
        ret = {}
        for label in labels:
            # 予測実施
            # ただし、sample_submitでは予測を全てrandomとしています。
            df[label] = df["code"].apply(lambda x: np.random.rand()) #cls.models[label].predict(feats)
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)
        
        return out.getvalue()