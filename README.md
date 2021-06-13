# [Signate JPX](https://signate.jp/competitions/423)
This repository contains the solution from team HODL from the Signate Tokyo Stock Exchange Fundamentals Modeling Competition. 

## Setup Guide
 1. Create conda environment `signate_jpx` with `environment.yml` file.
    ```bash
    conda env create -f environment.yml
    ```
 2. Activate `signate_jpx` conda env
    ```bash
    conda activate signate_jpx
    ```

To update environment. 
 ```bash
 conda env update --file environment.yml --prune
 ```

 3. Add csv.gz files to data/raw folder 

 4. To begin an end to end training ensure all data is in data/raw and run the following.
    
    ```bash
    sh start_pipeline.sh baseline_model 
    ```


## config.yml parameters 
The below contains a description of parameters in config.yml which guide the overall training pipeline. Depending on the setting of parameters, the behavior of the pipeline will change. 

* test_model: Whether to use public or private mode, Options: ["public", "private"]
* random_seed: random seed for models, Ex: 255
* low_memory_mode: Mode for debugging end to end pipeline quicker, uses only 10000 rows. Options: [True, False]
* use_fin_data: Whether to use the financial data, we have set to false as it did not help us much. Options: [True, False]
* data_date_limit: Maximum date for data,  Ex: "2021-01-01"
* train_split_date: Maximum date for training data  Ex: "2020-01-01"
* test_split_date: Starting date for test data Ex: "2020-01-01"
* drop_data: If we want to drop data from train and test --> Avoid Data leak -> drop 2 months data at end of dates Options: [True, False]
* drop_data_train_date: Drop data before this date. Ex: "2016-02-01"
* drop_data_test_date: Drop test data after this date. Ex: "2020-01-01"
* cross_validation: For internal validation, performs cross validation when training. Options: [True, False]
* lgb_model: Whether to use LightGBM model.  Options: [True]
* use_test_as_validation: Whether to use the test data as validation set. Options: [True, False]
* train_with_all_data: Used for training with all data, this was used for final submission as using more data is beneficial for improving model performance. Options: [True, False]
* lgb_params: LightGBM model hyperparameters