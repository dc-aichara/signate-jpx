# Automatically Copy utils to current 
cp -fr ../competition_code/utils ./src

# Automatically Copies Trained Models to submission
cp -fr ../models/lgb_label_high_20.txt  ./model
cp -fr ../models/lgb_label_low_20.txt  ./model

# Automatically Copy Data to directory 
cp -fr ../data/raw ./data

# Copy Configuration 
cp -fr ../config.yml ./src

# Copy Pre-Proc Objects/Metadata
cp -fr ../models/ordenc.pkl ./model
cp -fr ../models/scaler.pkl ./model
cp -fr ../models/metadata.json ./model