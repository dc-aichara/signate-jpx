## Setup Guide
How to setup docker in exactly the way they want for testing. 

### Build  
```docker build signate_jpx_serving```

### Start Container 
```docker-compose up```

### Enter Container 
```docker exec -it signate_jpx_serving /bin/bash```

### Testing 
The below functions are all what we must ensure are working for our submission. 

First type python to start using python commands. \
```python ```

path tbd

```DATASET_DIR= "/path/to"```

```from predictor import ScoringService```

```inputs = ScoringService.get_inputs(DATASET_DIR) ```

```ScoringService.get_model()```

```ScoringService.predict(inputs)```