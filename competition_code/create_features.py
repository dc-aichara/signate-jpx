import argparse
from utils import (
    load_config,
    get_data_rules,
    auto_categorical,
    auto_numeric,
    auto_dates,
    get_technical_features,
)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import json
import pickle


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help="Configuration ID for training"
    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Create Feature!!!!!")
    config = load_config(config_id)

    tree_model_data_config = {
        "base_date": "date",
        "Name (English)": "drop",
        "Section/Products": "categorical",
        "33 Sector(Code)": "categorical",
        "33 Sector(name)": "drop",
        "17 Sector(Code)": "categorical",
        "17 Sector(name)": "drop",
        "Size Code (New Index Series)": "categorical",
        "Size (New Index Series)": "drop",
        "IssuedShareEquityQuote AccountingStandard": "categorical",
        "IssuedShareEquityQuote IssuedShare": "numeric",
        "EndOfDayQuote Open": "numeric",
        "EndOfDayQuote High": "numeric",
        "EndOfDayQuote Low": "numeric",
        "EndOfDayQuote Close": "numeric",
        "EndOfDayQuote ExchangeOfficialClose": "numeric",
        "EndOfDayQuote Volume": "numeric",
        "EndOfDayQuote CumulativeAdjustmentFactor": "numeric",
        "EndOfDayQuote PreviousClose": "numeric",
        "EndOfDayQuote PreviousCloseDate": "drop",
        "EndOfDayQuote PreviousExchangeOfficialClose": "numeric",
        "EndOfDayQuote PreviousExchangeOfficialCloseDate": "drop",
        "EndOfDayQuote ChangeFromPreviousClose": "numeric",
        "EndOfDayQuote PercentChangeFromPreviousClose": "numeric",
        "EndOfDayQuote VWAP": "numeric",
    }

    # Generic Steps
    train = pd.read_csv("data/interim/train_data.csv")
    print("Train shape: ", train.shape)

    # Technical features for train data
    train_feat1 = get_technical_features(
        train, periods=[10, 20, 30], extra_feats=True
    )
    train = pd.merge(
        train, train_feat1, on=["base_date", "Local Code"], how="left"
    )

    tree_model_data_config.update(
        {col: "numeric" for col in train_feat1.columns[2:]}
    )
    del train_feat1

    train_feat2 = get_technical_features(train, periods=[14, 28, 42])

    train = pd.merge(
        train, train_feat2, on=["base_date", "Local Code"], how="left"
    )

    tree_model_data_config.update(
        {col: "numeric" for col in train_feat2.columns[2:]}
    )
    del train_feat2
    # New feats
    train["change_pct"] = (
        (train["EndOfDayQuote High"] - train["EndOfDayQuote Low"])
        * 100
        / (train["EndOfDayQuote Close"])
    )
    train["change_pct"] = train["change_pct"].fillna(0)
    train["change"] = train["EndOfDayQuote Open"] - train["EndOfDayQuote Close"]

    tree_model_data_config.update(
        {"change_pct": "numeric", "change": "numeric"}
    )

    if config.get("test_model") == "public":
        test = pd.read_csv("data/interim/test_data.csv")
        print("Test shape: ", test.shape)
        # Technical features for test data
        test_feat1 = get_technical_features(
            test, periods=[10, 20, 30], extra_feats=True
        )
        test = pd.merge(
            test, test_feat1, on=["base_date", "Local Code"], how="left"
        )
        del test_feat1

        test_feat2 = get_technical_features(test, periods=[14, 28, 42])
        test = pd.merge(
            test, test_feat2, on=["base_date", "Local Code"], how="left"
        )
        del test_feat2

        test["change_pct"] = (
            (test["EndOfDayQuote High"] - test["EndOfDayQuote Low"])
            * 100
            / (test["EndOfDayQuote Close"])
        )
        test["change_pct"] = test["change_pct"].fillna(0)
        test["change"] = (
            test["EndOfDayQuote Open"] - test["EndOfDayQuote Close"]
        )
        test.reset_index(drop=True, inplace=True)
        print("Test shape: ", test.shape)
    drop_data = config.get("drop_data")
    if drop_data:
        drop_data_train_date = config.get("drop_data_train_date")

        print("Drop date train:", drop_data_train_date)
        train = train[train["base_date"] >= drop_data_train_date]
        if config.get("test_model") == "public":
            drop_data_test_date = config.get("drop_data_test_date")
            print("Drop date test:", drop_data_test_date)
            test = test[test["base_date"] >= drop_data_test_date]
            test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)

    use_fin_data = config.get("use_fin_data")
    if use_fin_data:
        # Add fin columns to column type config dict
        fin_cols_config = {
            "Result_FinancialStatement AccountingStandard": "categorical",
            "Result_FinancialStatement ReportType": "categorical",
            "Result_FinancialStatement CompanyType": "categorical",
            "Result_FinancialStatement ChangeOfFiscalYearEnd": "categorical",
            "Result_FinancialStatement NetSales": "numeric",
            "Result_FinancialStatement OperatingIncome": "numeric",
            "Result_FinancialStatement OrdinaryIncome": "numeric",
            "Result_FinancialStatement NetIncome": "numeric",
            "Result_FinancialStatement TotalAssets": "numeric",
            "Result_FinancialStatement NetAssets": "numeric",
            "Forecast_FinancialStatement AccountingStandard": "categorical",
            "Forecast_FinancialStatement ReportType": "categorical",
            "Forecast_FinancialStatement NetSales": "numeric",
            "Forecast_FinancialStatement OperatingIncome": "numeric",
            "Forecast_FinancialStatement OrdinaryIncome": "numeric",
            "Forecast_FinancialStatement NetIncome": "numeric",
            "Result_Dividend ReportType": "categorical",
            "Result_Dividend QuarterlyDividendPerShare": "numeric",
            "Forecast_Dividend ReportType": "categorical",
            "Forecast_Dividend QuarterlyDividendPerShare": "numeric",
        }
        tree_model_data_config.update(fin_cols_config)

    # Split into X/y Train, X/Y test
    y_train_high = train["label_high_20"]
    y_train_low = train["label_low_20"]
    print(
        y_train_high.shape,
        y_train_low.shape,
        train.shape,
    )
    y_train_high.to_csv("data/processed/y_train_high.csv", index=False)
    y_train_low.to_csv("data/processed/y_train_low.csv", index=False)
    del y_train_high, y_train_low
    if config.get("test_model") == "public":
        y_test_high = test["label_high_20"]
        y_test_low = test["label_low_20"]
        y_test_high.to_csv("data/processed/y_test_high.csv", index=False)
        y_test_low.to_csv("data/processed/y_test_low.csv", index=False)
        del (
            y_test_low,
            y_test_high,
        )
    print("Saved Labels!!!")

    # Get preproc rules
    numerics_tree, dates_tree, categoricals_tree, drops_tree = get_data_rules(
        tree_model_data_config
    )

    # Trees
    print("Begin Tree Processing")

    train_dates_df_trees = auto_dates(train, dates_tree)
    ordenc = OrdinalEncoder()
    ordenc.fit(train[categoricals_tree])
    train_categoricals_df_trees = auto_categorical(
        train, ordenc, categoricals_tree
    )
    train_numerics_df_trees = auto_numeric(train, scaler, numerics_tree)

    train_trees = pd.concat(
        [
            train_dates_df_trees,
            train_categoricals_df_trees,
            train_numerics_df_trees,
        ],
        axis=1,
    )

    train_trees.to_csv("data/processed/train_trees.csv", index=False)
    cols = train_trees.columns.tolist()
    del train_trees
    if config.get("test_model") == "public":
        # Only for our own evaluation purposes
        # Trees
        test_dates_df_trees = auto_dates(test, dates_tree)
        test_numerics_df_trees = auto_numeric(test, scaler, numerics_tree)
        test_categoricals_df_trees = auto_categorical(
            test, ordenc, categoricals_tree
        )
        test_trees = pd.concat(
            [
                test_dates_df_trees,
                test_categoricals_df_trees,
                test_numerics_df_trees,
            ],
            axis=1,
        )
        print("Test tree data shape", test_trees.shape)
        test_trees.to_csv("data/processed/test_trees.csv", index=False)

    # Save Model Objects
    scaler_output = open("models/scaler.pkl", "wb")
    pickle.dump(scaler, scaler_output)

    ordenc_output = open("models/ordenc.pkl", "wb")
    pickle.dump(ordenc, ordenc_output)

    # Create Metadata for serving
    metadata = {
        "categorical": categoricals_tree,
        "dates": dates_tree,
        "numeric": numerics_tree,
        "col_order": cols,
    }

    print(metadata)

    with open("models/metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
