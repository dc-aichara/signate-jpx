import argparse
import pandas as pd
from utils import load_config
from utils import time_series_CV





if __name__ == '__main__':
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id", type=str, help='Configuration ID for training'

    )

    ARGS = CLI.parse_args()

    config_id = ARGS.config_id

    print('Make Dataset !!!!')
    config = load_config(config_id)


    stock_labels = pd.read_csv('data/raw/stock_labels.csv')
    stock_fin_price = pd.read_csv('data/raw/stock_fin_price.csv')
    stock_fin = pd.read_csv('data/raw/stock_fin.csv')
    stock_list = pd.read_csv('data/raw/stock_list.csv')
    stock_price = pd.read_csv('data/raw/stock_price.csv')

    print(stock_labels.shape)

    if config.get('test_model') == 'public': 
        print('Removing Data as model will only be used on test set')
        train = stock_labels[stock_labels['base_date'] < '2020-01-01']
        test = stock_labels[stock_labels['base_date'] >= '2020-01-01']
        print(train.shape)

    else: 
        train = stock_labels   


    train = pd.merge(train, stock_list, on=['Local Code'], how='left')
    print(train.shape)
    train = pd.merge(train, stock_fin_price, on=['base_date','Local Code'], how='left')
    print(train.shape)
    train = pd.merge(train, stock_fin, on=['base_date','Local Code'], how='left')
    print(train.shape)
    train = pd.merge(train, stock_price, left_on=['base_date', 'Local Code'], right_on=['EndOfDayQuote Date', 'Local Code'], how='left')


    print(train.shape)
    print(train.head())
    ## Begin combining data and stuff. 


    ## Write files to interim 

    ## 

