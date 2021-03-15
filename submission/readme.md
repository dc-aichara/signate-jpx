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

Run submission testing script to test submission

```bash
cd src
python test_submission.py
```


