## Setup Guide
How to set up docker in exactly the way they want for testing. 

Before you start you need to run the shell script which will do all the preparatory work for our submission. This script will move various files needed to test.
```bash
cd submission
```
```bash
sh prepare_submission.sh
```

### Build  
```bash
docker-compose build
```

### Start Container 
```bash
docker-compose up
```

### Enter Container 
```bash
docker exec -it signate_jpx_serving /bin/bash
```

### Testing 
The below functions are all what we must ensure are working for our submission. 

There is a test script called test_submission.py which can be used to activate testing. 
```bash
python test_submission.py
```

