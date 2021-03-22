from datetime import datetime

t1 = datetime.now()
DATASET_DIR = "../data"

from predictor import ScoringService

# Gets original datasets
inputs = ScoringService.get_inputs(DATASET_DIR)
assert inputs == {'stock_list': '../data/stock_list.csv.gz',
                  'stock_price': '../data/stock_price.csv.gz',
                  'stock_labels': '../data/stock_labels.csv.gz'}

# Reads Dataframes into CSVs
datasets = ScoringService.get_dataset(inputs)
print(datasets)

codes = ScoringService.get_codes(datasets)
print(codes)

model = ScoringService.get_model()

assert model == True 

predictions = ScoringService.predict(inputs)
t2 = datetime.now()
print(predictions)
print(f"Total time: {(t2 - t1).seconds} seconds")
