# Convert to pyodide
#
panel convert index.py --to pyodide-worker --out docs/

# Add in the header information we need.
# This includes unpacking the zipped data and 
# adding the google analytics tag.
#
python bundle.py

# python process_data.py
# zip docs/processed_data.zip processed_data/