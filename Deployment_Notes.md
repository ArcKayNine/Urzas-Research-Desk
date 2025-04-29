# Deployment Notes
In order to deploy there are a few steps:
- Make sure MTGODecklistCache is updated.
- Run process_data.py in the Processing repo.
- Zip up the result and put it in docs/.
- Run bundle.sh to add required headers.

# Testing
To test, run

```
python -m http.server
```

and navigate to 127.0.0.1:8000

# Running in Python
The only difference above is to have the data sitting in processed_data. 
Then use the load_data function from FormatAnalysis.py