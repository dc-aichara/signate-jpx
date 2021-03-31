import argparse
import pandas as pd
from utils import load_config
from utils import load_data
from utils import format_dates
from datetime import datetime


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help="Configuration ID for training"
    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print("Make Dataset !!!!")
    config = load_config(config_id)
    use_fin_data = config.get("use_fin_data")
    train_split_date = config.get("train_split_date", "2020-01-01")
    test_split_date = config.get("test_split_date", "2020-01-01")
    time1 = datetime.now()
    (
        stock_price,
        stock_fin,
        stock_list,
        stock_labels,
    ) = load_data()
    time2 = datetime.now()

    # Fix all Date Formats
    stock_price = format_dates(stock_price, ["EndOfDayQuote Date"])
    stock_fin = format_dates(stock_fin, ["base_date"])
    stock_price.rename(
        columns={"EndOfDayQuote Date": "base_date"}, inplace=True
    )
    # Drop data by date limit
    data_date_limit = config.get("data_date_limit", "2021-01-01")
    stock_price = stock_price[stock_price["base_date"] < data_date_limit]
    # drop columns with greater than 20% null values
    stock_fin.drop(
        stock_fin.isnull()
        .sum()[stock_fin.isnull().sum() / len(stock_fin) * 100 > 20]
        .index,
        axis=1,
        inplace=True,
    )
    # Fill NAs
    stock_fin.fillna(method="ffill", inplace=True)
    stock_fin.fillna(method="bfill", inplace=True)

    if use_fin_data:
        stock_price = pd.merge(
            stock_price, stock_fin, on=["base_date", "Local Code"], how="left"
        )
        stock_price.fillna(method="ffill", inplace=True)
        print("NAs percentage in each columns after merge and NA fill.")
        print(stock_price.isnull().sum() / len(stock_price) * 100)

    stock_labels = format_dates(
        stock_labels,
        ["base_date", "label_date_5", "label_date_10", "label_date_20"],
    )

    print(stock_price.shape)
    if config.get("test_model") == "public":
        print("Removing Data as model will only be used on test_data set")
        train = stock_price[stock_price["base_date"] < train_split_date]
        test = stock_price[stock_price["base_date"] >= test_split_date]
        test.reset_index(drop=True, inplace=True)
        print(train.shape, test.shape)

    else:
        train = stock_price

    if config.get("low_memory_mode") is True:
        train = train[0:10000]
        test = test[0:10000]

    train = pd.merge(train, stock_list, on=["Local Code"], how="left")
    print(train.shape)

    train = train[train["prediction_target"] == True]

    train = pd.merge(
        train,
        stock_labels,
        on=["base_date", "Local Code"],
        how="left",
    )
    # Drop first few rows of train data which has NAs.
    if use_fin_data:
        train.dropna(subset=["Forecast_Dividend FiscalPeriodEnd"], inplace=True)
    train.reset_index(inplace=True, drop=True)

    # Handle Test

    if config.get("test_model") == "public":
        test = pd.merge(test, stock_list, on=["Local Code"], how="left")
        test = test[test["prediction_target"] == True]

        test = pd.merge(
            test,
            stock_labels,
            on=["base_date", "Local Code"],
            how="left",
        )

    time3 = datetime.now()

    print(train.shape)
    print(train.head())
    print(f"Data Loading time {time2 - time1}")
    print(f"Data merging time {time3 - time2}")
    print(train.isnull().sum() / len(train) * 100)
    # Begin combining data and stuff.
    train["33 Sector(Code)"] = train["33 Sector(Code)"].astype(int)
    train["17 Sector(Code)"] = train["17 Sector(Code)"].astype(int)
    train.to_csv("data/interim/train_data.csv", index=False)

    if config.get("test_model") == "public":
        test["33 Sector(Code)"] = test["33 Sector(Code)"].astype(int)
        test["17 Sector(Code)"] = test["17 Sector(Code)"].astype(int)
        test.to_csv("data/interim/test_data.csv", index=False)
        print(test.shape)
        print(test.head())
