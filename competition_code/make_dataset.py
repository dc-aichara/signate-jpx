import argparse
import pandas as pd
from utils import load_config
from utils import time_series_CV
from utils import reduce_mem_usage
from datetime import datetime


def load_data(data_dir: str = "data/raw/"):
    print(f"Loading data from {data_dir}")
    stock_labels = pd.read_csv(f"{data_dir}stock_labels.csv.gz")
    stock_fin_price = pd.read_csv(f"{data_dir}stock_fin_price.csv.gz")
    stock_fin = pd.read_csv(f"{data_dir}stock_fin.csv.gz")
    stock_list = pd.read_csv(f"{data_dir}stock_list.csv.gz")
    stock_price = pd.read_csv(f"{data_dir}stock_price.csv.gz")

    return stock_price, stock_fin_price, stock_fin, stock_list, stock_labels


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
    time1 = datetime.now()
    (
        stock_price,
        stock_fin_price,
        stock_fin,
        stock_list,
        stock_labels,
    ) = load_data()

    stock_price = reduce_mem_usage(stock_price)
    stock_fin_price = reduce_mem_usage(stock_fin_price)
    stock_fin = reduce_mem_usage(stock_fin)
    stock_list = reduce_mem_usage(stock_list)
    stock_labels = reduce_mem_usage(stock_labels)

    # Fix all Date Formats
    stock_price = format_dates(stock_price, ["EndOfDayQuote Date"])

    # stock_fin_price = format_dates(stock_fin_price, ['base_date',
    #                                                 'EndOfDayQuote PreviousCloseDate',
    #                                                 'EndOfDayQuote PreviousExchangeOfficialCloseDate',
    #                                                 'EndOfDayQuote PreviousCloseDate',
    #                                                 'EndOfDayQuote PreviousExchangeOfficialCloseDate'])
    # stock_fin = format_dates(stock_fin, ['base_date',
    #                                     'Result_FinancialStatement ModifyDate',
    #                                     'Forecast_FinancialStatement ModifyDate',
    #                                     'Forecast_Dividend ModifyDate',
    #                                     'Forecast_Dividend RecordDate'])
    # stock_list = format_dates(stock_list,['IssuedShareEquityQuote ModifyDate'])

    stock_labels = format_dates(
        stock_labels,
        ["base_date", "label_date_5", "label_date_10", "label_date_20"],
    )

    print(stock_labels.shape)
    time2 = datetime.now()
    if config.get("test_model") == "public":
        print("Removing Data as model will only be used on test set")
        train = stock_labels[stock_labels["base_date"] < "2020-01-01"]
        test = stock_labels[stock_labels["base_date"] >= "2020-01-01"]
        print(train.shape)

    else:
        train = stock_labels

    if config.get("low_memory_mode") is True:
        train = train[0:10000]
        test = test[0:10000]

    train = pd.merge(train, stock_list, on=["Local Code"], how="left")
    print(train.shape)
    # train = pd.merge(
    #    train, stock_fin_price, on=["base_date", "Local Code"], how="left"
    # )

    test = pd.merge(test, stock_list, on=["Local Code"], how="left")
    print(train.shape)
    # test = pd.merge(
    # test, stock_fin_price, on=["base_date", "Local Code"], how="left")

    train = train[train["prediction_target"] == True]
    test = test[test["prediction_target"] == True]

    print(train.shape)
    # train = pd.merge(
    #    train, stock_fin, on=["base_date", "Local Code"], how="left"
    # )
    # print(train.shape)
    train = pd.merge(
        train,
        stock_price,
        left_on=["base_date", "Local Code"],
        right_on=["EndOfDayQuote Date", "Local Code"],
        how="left",
    )

    test = pd.merge(
        test,
        stock_price,
        left_on=["base_date", "Local Code"],
        right_on=["EndOfDayQuote Date", "Local Code"],
        how="left",
    )

    time3 = datetime.now()

    print(train.shape)
    print(train.head())
    print(f"Data Loading time {time2 - time1}")
    print(f"Data merging time {time3 - time2}")

    ## Begin combining data and stuff.
    train.to_csv("data/interim/train.csv")
    test.to_csv("data/interim/test.csv")
