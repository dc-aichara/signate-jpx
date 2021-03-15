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

    # Probably will move this - Anything we want to keep
    linear_model_data_config = {
        "base_date": "date",
        # 'Effective Date': 'date',
        # 'Local Code': 'categorical',
        "Name (English)": "drop",
        "Section/Products": "categorical",
        "33 Sector(Code)": "categorical",
        "33 Sector(name)": "drop",
        "17 Sector(Code)": "categorical",
        "17 Sector(name)": "drop",
        "Size Code (New Index Series)": "categorical",
        "Size (New Index Series)": "drop",
        "IssuedShareEquityQuote AccountingStandard": "categorical",
        # 'IssuedShareEquityQuote ModifyDate': 'date',
        "IssuedShareEquityQuote IssuedShare": "numeric",
        # stock_price
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

    tree_model_data_config = {
        "base_date": "date",
        # 'Effective Date': 'date',
        # 'Local Code': 'categorical',
        "Name (English)": "drop",
        "Section/Products": "categorical",
        "33 Sector(Code)": "categorical",
        "33 Sector(name)": "drop",
        "17 Sector(Code)": "categorical",
        "17 Sector(name)": "drop",
        "Size Code (New Index Series)": "categorical",
        "Size (New Index Series)": "drop",
        "IssuedShareEquityQuote AccountingStandard": "categorical",
        # 'IssuedShareEquityQuote ModifyDate': 'date',
        "IssuedShareEquityQuote IssuedShare": "numeric",
        # stock_price
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
    test = pd.read_csv("data/interim/test_data.csv")
    print(train.shape, test.shape)

    # Get simple Technical features
    train_feat1 = get_technical_features(
        train, periods=[10, 20, 30], extra_feats=True
    )
    test_feat1 = get_technical_features(
        test, periods=[10, 20, 30], extra_feats=True
    )
    print(train_feat1.columns)

    train = pd.merge(
        train, train_feat1, on=["base_date", "Local Code"], how="left"
    )
    test = pd.merge(
        test, test_feat1, on=["base_date", "Local Code"], how="left"
    )
    linear_model_data_config.update(
        {col: "numeric" for col in train_feat1.columns[2:]}
    )
    tree_model_data_config.update(
        {col: "numeric" for col in train_feat1.columns[2:]}
    )
    del train_feat1, test_feat1
    train_feat2 = get_technical_features(train, periods=[14, 28, 42])
    test_feat2 = get_technical_features(test, periods=[14, 28, 42])

    train = pd.merge(
        train, train_feat2, on=["base_date", "Local Code"], how="left"
    )
    test = pd.merge(
        test, test_feat2, on=["base_date", "Local Code"], how="left"
    )

    linear_model_data_config.update(
        {col: "numeric" for col in train_feat2.columns[2:]}
    )
    tree_model_data_config.update(
        {col: "numeric" for col in train_feat2.columns[2:]}
    )
    del train_feat2, test_feat2
    print(train.shape, test.shape)
    drop_data = config.get("drop_data")
    if drop_data:
        drop_data_train_date = config.get("drop_data_train_date")
        drop_data_test_date = config.get("drop_data_test_date")
        print("Drop dates:", drop_data_test_date, drop_data_train_date)
        train = train[train["base_date"] >= drop_data_train_date]
        test = test[test["base_date"] >= drop_data_test_date]
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    print(train.shape, test.shape)

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
        linear_model_data_config.update(fin_cols_config)
        tree_model_data_config.update(fin_cols_config)

    # Split into X/y Train, X/Y test
    y_train_high = train["label_high_20"]
    y_train_low = train["label_low_20"]

    y_test_high = test["label_high_20"]
    y_test_low = test["label_low_20"]

    print(
        y_train_high.shape,
        y_train_low.shape,
        y_test_high.shape,
        y_test_low.shape,
        train.shape,
        test.shape,
    )
    y_train_high.to_csv("data/processed/y_train_high.csv", index=False)
    y_train_low.to_csv("data/processed/y_train_low.csv", index=False)
    y_test_high.to_csv("data/processed/y_test_high.csv", index=False)
    y_test_low.to_csv("data/processed/y_test_low.csv", index=False)
    print("Saved Labels!!!")
    del y_test_low, y_test_high, y_train_high, y_train_low
    # Get preproc rules
    (
        numerics_linear,
        dates_linear,
        categoricals_linear,
        drops_linear,
    ) = get_data_rules(linear_model_data_config)
    numerics_tree, dates_tree, categoricals_tree, drops_tree = get_data_rules(
        tree_model_data_config
    )

    # Linear
    print("Begin Linear Processing")

    # Dates
    train_dates_df_linear = auto_dates(train, dates_linear)

    # Numerics
    scaler = MinMaxScaler()
    scaler.fit(train[numerics_linear])

    train_numerics_df_linear = auto_numeric(train, scaler, numerics_linear)

    # Categoricals
    for category in categoricals_linear:
        train[category] = train[category].fillna("no_category").astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(train[categoricals_linear])
    train_categoricals_df_linear = auto_categorical(
        train, ohe, categoricals_linear
    )

    train_linear = pd.concat(
        [
            train_dates_df_linear,
            train_categoricals_df_linear,
            train_numerics_df_linear,
        ],
        axis=1,
    )

    train_linear.to_csv("data/processed/train_linear.csv", index=False)
    del train_linear
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
    del train_trees
    if config.get("test_model") == "public":
        # Only for our own evaluation purposes

        # Linear
        test_dates_df_linear = auto_dates(test, dates_linear)
        test_numerics_df_linear = auto_numeric(test, scaler, numerics_linear)
        test_categoricals_df_linear = auto_categorical(
            test, ohe, categoricals_linear
        )
        test_linear = pd.concat(
            [
                test_dates_df_linear,
                test_categoricals_df_linear,
                test_numerics_df_linear,
            ],
            axis=1,
        )
        print("Test linear data shape", test_linear.shape)
        test_linear.to_csv("data/processed/test_linear.csv", index=False)

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
            "col_order": test_trees.columns.to_list(),
        }

        print(metadata)

        with open("models/metadata.json", "w") as fp:
            json.dump(metadata, fp, indent=2)
