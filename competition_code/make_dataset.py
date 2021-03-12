import argparse
import pandas as pd
from utils import load_config
from utils import time_series_CV
from utils import reduce_mem_usage
from datetime import datetime


def load_data(data_dir: str = "data/raw/"):
    print(f"Loading data from {data_dir}")
    stock_labels = pd.read_csv(f"{data_dir}stock_labels.csv.gz")
    stock_fin = pd.read_csv(f"{data_dir}stock_fin.csv.gz")
    stock_list = pd.read_csv(f"{data_dir}stock_list.csv.gz")
    stock_price = pd.read_csv(f"{data_dir}stock_price.csv.gz")

    return stock_price, stock_fin, stock_list, stock_labels


def format_dates(df: pd.DataFrame, columns: list):

    for column in columns:
        df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d")

    return df


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
    # stock_price = reduce_mem_usage(stock_price)
    # stock_fin = reduce_mem_usage(stock_fin)
    # stock_list = reduce_mem_usage(stock_list)
    # stock_labels = reduce_mem_usage(stock_labels)

    # Fix all Date Formats
    stock_price = format_dates(stock_price, ["EndOfDayQuote Date"])
    stock_fin = format_dates(stock_fin, ["base_date"])
    stock_price.rename(
        columns={"EndOfDayQuote Date": "base_date"}, inplace=True
    )

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

    print(stock_labels.shape)
    if config.get("test_model") == "public":
        print("Removing Data as model will only be used on test_data set")
        # train = stock_labels[stock_labels["base_date"] < "2020-01-01"]
        # test = stock_labels[stock_labels["base_date"] >= "2020-02-01"]
        train = stock_labels[stock_labels["base_date"] < train_split_date]
        test = stock_labels[stock_labels["base_date"] >= test_split_date]
        test.reset_index(drop=True, inplace=True)
        print(train.shape, test.shape)

    else:
        train = stock_labels

    if config.get("low_memory_mode") is True:
        train = train[0:10000]
        test = test[0:10000]

    train = pd.merge(train, stock_list, on=["Local Code"], how="left")
    print(train.shape)

    train = train[train["prediction_target"] == True]

    train = pd.merge(
        train,
        stock_price,
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
            stock_price,
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
    train.to_csv("data/interim/train_data.csv", index=False)

    if config.get("test_model") == "public":
        test.to_csv("data/interim/test_data.csv", index=False)
        print(test.shape)
        print(test.head())
