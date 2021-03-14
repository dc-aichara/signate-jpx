## Setup Guide
How to setup docker in exactly the way they want for testing. 

Before you start you need to run the shell script which will do all the prepatory work for our submission. This script will move various files needed to test. 
```sh prepare_submission.sh```

### Build  
```docker build signate_jpx_serving```

### Start Container 
```docker-compose up```

### Enter Container 
```docker exec -it signate_jpx_serving /bin/bash```

### Testing 
The below functions are all what we must ensure are working for our submission. 

There is a test script called test_submission.py which can be used to activate testing. 
```python test_submission.py```

