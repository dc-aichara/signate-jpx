import argparse
import pandas as pd
from utils import load_config
from utils import time_series_CV
from datetime import datetime


def load_data(data_dir: str = "data/raw/"):
    print(f"Loading data from {data_dir}")
    stock_labels = pd.read_csv(f"{data_dir}stock_labels.csv.gz")
    stock_fin_price = pd.read_csv(f"{data_dir}stock_fin_price.csv.gz")
    stock_fin = pd.read_csv(f"{data_dir}stock_fin.csv.gz")
    stock_list = pd.read_csv(f"{data_dir}stock_list.csv.gz")
    stock_price = pd.read_csv(f"{data_dir}stock_price.csv.gz")

    return stock_price, stock_fin_price, stock_fin, stock_list, stock_labels


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
    print(stock_labels.shape)
    time2 = datetime.now()
    if config.get("test_model") == "public":
        print("Removing Data as model will only be used on test set")
        train = stock_labels[stock_labels["base_date"] < "2020-01-01"]
        test = stock_labels[stock_labels["base_date"] >= "2020-01-01"]
        print(train.shape)

    else:
        train = stock_labels

    train = pd.merge(train, stock_list, on=["Local Code"], how="left")
    print(train.shape)
    train = pd.merge(
        train, stock_fin_price, on=["base_date", "Local Code"], how="left"
    )
    print(train.shape)
    train = pd.merge(
        train, stock_fin, on=["base_date", "Local Code"], how="left"
    )
    print(train.shape)
    train = pd.merge(
        train,
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

    ## Write files to interim

    ##
