import argparse
from utils import load_config, calculate_price_indices, date_feats
from dateutil import parser
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


def get_data_rules(config: dict):
    numerics = []
    dates = []
    categoricals = []
    drops = []

    for key, value in config.items():
        if value == "numeric":
            numerics.append(key)
        if value == "date":
            dates.append(key)
        if value == "categorical":
            categoricals.append(key)
        if value == "drop":
            drops.append(key)

    return numerics, dates, categoricals, drops


def auto_categorical(df, encoder, categoricals):
    for category in categoricals:
        df[category] = df[category].fillna("no_category").astype(str)

    transformed_df = pd.DataFrame(encoder.transform(df[categoricals]))

    # if ohe
    if isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder):
        print("OHE")
        transformed_df.columns = encoder.get_feature_names()

    if isinstance(encoder, sklearn.preprocessing._encoders.OrdinalEncoder):
        print("Ordinal Encoder")
        transformed_df.columns = categoricals

    return transformed_df


def auto_numeric(df, scaler, numerics):
    numerics_df = pd.DataFrame(scaler.transform(df[numerics]))
    numerics_df.columns = numerics

    return numerics_df


def auto_dates(train, dates):
    # DF with all the dates
    dates_df = pd.DataFrame()
    for date in dates:
        dates_df[f"{date}_month"] = pd.to_datetime(train[date]).dt.month
        dates_df[f"{date}_day"] = pd.to_datetime(train[date]).dt.day
        dates_df[f"{date}_dayofweek"] = pd.to_datetime(train[date]).dt.dayofweek

    return dates_df


def get_technical_features(
    df,
    date_col="base_date",
    price_col="EndOfDayQuote ExchangeOfficialClose",
    periods=[7, 14, 21],
):
    """
    Args:
        df (pd.DataFrame): DataFrame
        date_col (str): Date column in DataFrame
        price_col (str): Price column in DataFrame
        periods (list): List of periods to create technical features
    Returns:
        pd.DataFrame: Feature DataFrame
    """
    data = df[["Local Code", date_col, price_col]]
    datas = []
    for code in data["Local Code"].unique():
        feats = data[data["Local Code"] == code]
        feats[f"return_{periods[0]}"] = feats[price_col].pct_change(periods[0])
        feats[f"return_{periods[1]}"] = feats[price_col].pct_change(periods[1])
        feats[f"return_{periods[2]}"] = feats[price_col].pct_change(periods[2])
        feats[f"volatility_{periods[0]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[0]).std()
        )
        feats[f"volatility_{periods[1]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[2]).std()
        )
        feats[f"volatility_{periods[2]}"] = (
            np.log(feats[price_col]).diff().rolling(periods[2]).std()
        )
        feats[f"MA_gap_{periods[0]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[0]).mean()
        )
        feats[f"MA_gap_{periods[1]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[1]).mean()
        )
        feats[f"MA_gap_{periods[2]}"] = feats[price_col] / (
            feats[price_col].rolling(periods[2]).mean()
        )
        feats = feats.fillna(0)
        feats = feats.drop([price_col], axis=1)
        datas.append(feats)
    feats = pd.concat(datas, ignore_index=True)
    del datas, data
    return feats


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
        ####### stock_price
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
        ####### stock_price
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
    train_feat1 = get_technical_features(train, periods=[10, 20, 30])
    test_feat1 = get_technical_features(test, periods=[10, 20, 30])
    train_feat2 = get_technical_features(train, periods=[14, 28, 42])
    test_feat2 = get_technical_features(test, periods=[14, 28, 42])

    train = pd.merge(
        train_feat1, train, on=["base_date", "Local Code"], how="left"
    )
    test = pd.merge(
        test_feat1, test, on=["base_date", "Local Code"], how="left"
    )

    train = pd.merge(
        train_feat2, train, on=["base_date", "Local Code"], how="left"
    )
    test = pd.merge(
        test_feat2, test, on=["base_date", "Local Code"], how="left"
    )

    linear_model_data_config.update(
        {col: "numeric" for col in train_feat1.columns[2:]}
    )
    tree_model_data_config.update(
        {col: "numeric" for col in train_feat1.columns[2:]}
    )

    linear_model_data_config.update(
        {col: "numeric" for col in train_feat2.columns[2:]}
    )
    tree_model_data_config.update(
        {col: "numeric" for col in train_feat2.columns[2:]}
    )

    # Add fin columns to column type config dict
    linear_model_data_config.update(
        {
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
    )
    tree_model_data_config.update(
        {
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
    )
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
    )
    y_train_high.to_csv("data/processed/y_train_high.csv", index=False)
    y_train_low.to_csv("data/processed/y_train_low.csv", index=False)
    y_test_high.to_csv("data/processed/y_test_high.csv", index=False)
    y_test_low.to_csv("data/processed/y_test_low.csv", index=False)
    print("Saved Labels!!!")
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
        test_trees.to_csv("data/processed/test_trees.csv", index=False)
