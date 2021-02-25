# Signate JPX


# Setup Guide

## Development Env
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

## DVC 
 1. To start dvc run the following at the beginning of project. 
      ```bash
    dvc init
    ```
 2. Anytime you train a model be sure to use the start_pipeline.sh script and specify correct inputs and outputs. 

### Docker Setup - For running pipeline 
 1. Build Docker environment from conda  
     
    ```bash
    docker-compose build 
    ```
 2. Run docker container- should start up jupyter automatically
      
    ```bash
    docker-compose up 
    ```
 3. To utilize jupyter environment click the link in the log to be taken to jupyter environment with conda pre-loaded. 
 4. To utilize training pipeline run the following command. 
    
    ```bash
    docker exec -it signate_jpx_pipeline /bin/bash 
    ```
 5. To begin an end to end training ensure all data is in data/raw and run the following. - maybe need to allow for loading data from old files in future.
    
    ```bash
    sh start_pipeline.sh 
    ```
### Data Setup 
 1. Add CSV files to data/raw.


