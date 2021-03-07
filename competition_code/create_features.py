import argparse
from utils import load_config, calculate_price_indices, date_feats
from dateutil import parser
import pandas as pd
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
        dates_df["f{date}_dayofweek"] = pd.to_datetime(train[date]).dt.dayofweek

    return dates_df


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
        #'Effective Date': 'date',
        #'Local Code': 'categorical',
        "Name (English)": "drop",
        "Section/Products": "categorical",
        "33 Sector(Code)": "categorical",
        "33 Sector(name)": "drop",
        "17 Sector(Code)": "categorical",
        "17 Sector(name)": "drop",
        "Size Code (New Index Series)": "categorical",
        "Size (New Index Series)": "drop",
        "IssuedShareEquityQuote AccountingStandard": "categorical",
        #'IssuedShareEquityQuote ModifyDate': 'date',
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

    ## Generic Steps
    train = pd.read_csv("data/interim/train.csv")
    test = pd.read_csv("data/interim/test.csv")

    # Split into X/y Train, X/Y test
    y_train_high = train["label_high_20"]
    y_train_low = train["label_low_20"]

    y_test_high = test["label_high_20"]
    y_test_low = test["label_low_20"]

    y_train_high.to_csv("data/processed/y_train_high.csv", index=False)
    y_train_low.to_csv("data/processed/y_train_low.csv", index=False)
    y_test_high.to_csv("data/processed/y_test_high.csv", index=False)
    y_test_low.to_csv("data/processed/y_test_low.csv", index=False)

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
        test_trees.to_csv("data/processed/test_tress.csv", index=False)
