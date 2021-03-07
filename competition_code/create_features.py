import argparse
from utils import load_config, calculate_price_indices, date_feats
from dateutil import parser
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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


def auto_categorical(train, test, categoricals):
    for category in categoricals:
        train[category] = train[category].fillna("no_category").astype(str)
        test[category] = test[category].fillna("no_category").astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ohe.fit(train[categoricals])

    transformed_train = pd.DataFrame(ohe.transform(train[categoricals]))
    print(transformed_train.shape)
    print(len(ohe.get_feature_names()))
    transformed_train.columns = ohe.get_feature_names()

    transformed_test = pd.DataFrame(ohe.transform(test[categoricals]))
    transformed_test.columns = ohe.get_feature_names()

    ## Add Ordinal Encoder for LGBM later

    return transformed_train, transformed_test


def auto_numeric(train, test, numerics):
    scaler = MinMaxScaler()

    scaler.fit(train[numerics])
    train_numeric_df = pd.DataFrame(scaler.transform(train[numerics]))
    train_numeric_df.columns = numerics

    test_numeric_df = pd.DataFrame(scaler.transform(test[numerics]))
    test_numeric_df.columns = numerics

    return train_numeric_df, test_numeric_df


def auto_dates(train, test, dates):
    train_dates_df = pd.DataFrame()
    test_dates_df = pd.DataFrame()
    for date in dates:
        # try:
        #    df[date] = df[date].apply(lambda x: parser.parse(str(x)))
        # except:
        # df[date] = pd.to_datetime(df[date]).strftime("%Y-%m-%d")
        # df[date].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notnull(x) else '')
        train_dates = date_feats(
            train[[date]],
            date_col=date,
            month_col="month",
            dom_col="dayofmonth",
            dow_col="dayofweek",
        )
        (
            train_dates_df[f"{date}month"],
            train_dates_df[f"{date}_dayofmonth"],
            train_dates_df[f"{date}_dayofweek"],
        ) = (
            train_dates["month"],
            train_dates["dayofmonth"],
            train_dates["dayofweek"],
        )
        del train_dates
        test_dates = date_feats(
            test[[date]],
            date_col=date,
            month_col="month",
            dom_col="dayofmonth",
            dow_col="dayofweek",
        )
        (
            test_dates_df[f"{date}month"],
            test_dates_df[f"{date}_dayofmonth"],
            test_dates_df[f"{date}_dayofweek"],
        ) = (
            test_dates["month"],
            test_dates["dayofmonth"],
            test_dates["dayofweek"],
        )
        del test_dates

    return train_dates_df, test_dates_df


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
    data_processing_config = {
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

    numerics, dates, categoricals, drops = get_data_rules(
        data_processing_config
    )

    train = pd.read_csv("data/interim/train.csv")
    test = pd.read_csv("data/interim/test.csv")

    # Split into X/y Train, X/Y test
    y_train_high = train["label_high_20"]
    y_train_low = train["label_low_20"]

    y_test_high = test["label_high_20"]
    y_test_low = test["label_low_20"]

    print("Auto Train")
    train_dates_df, test_dates_df = auto_dates(train, test, dates)
    train_categoricals_df, test_categoricals_df = auto_categorical(
        train, test, categoricals
    )
    train_numerics_df, test_numerics_df = auto_numeric(train, test, numerics)

    train = pd.concat(
        [train_dates_df, train_categoricals_df, train_numerics_df], axis=1
    )
    test = pd.concat(
        [test_dates_df, test_categoricals_df, test_numerics_df], axis=1
    )

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    y_train_high.to_csv("data/processed/y_train_high.csv", index=False)
    y_train_low.to_csv("data/processed/y_train_low.csv", index=False)
    y_test_high.to_csv("data/processed/y_test_high.csv", index=False)
    y_test_low.to_csv("data/processed/y_test_low.csv", index=False)
