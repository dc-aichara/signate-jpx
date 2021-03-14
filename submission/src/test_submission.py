

DATASET_DIR= "../data/raw"

from predictor import ScoringService

## Gets original datasets
inputs = ScoringService.get_inputs(DATASET_DIR)
assert inputs == {'stock_list': '../data/raw/stock_list.csv.gz', 'stock_price': '../data/raw/stock_price.csv.gz', 'stock_labels': '../data/raw/stock_labels.csv.gz'}

# Reads Dataframes into CSVs
datasets= ScoringService.get_dataset(inputs)
print(datasets)

codes = ScoringService.get_codes(datasets)
print(codes)

model = ScoringService.get_model()

assert model == True 

predictions = ScoringService.predict(inputs)

print(predictions)