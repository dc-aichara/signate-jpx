# Competition Pipeline
config_id=$1
echo $config_id 

python ./competition_code/make_dataset.py --config_id ${config_id}
python ./competition_code/create_features.py --config_id ${config_id}
python ./competition_code/train_model.py --config_id ${config_id}
python ./competition_code/model_serving.py --config_id ${config_id}