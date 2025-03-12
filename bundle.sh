# Run the processing and save in a zip.
#
echo "Processing data..."
formats=(
    Standard
    Pioneer
    Modern
    Legacy
    Vintage
)
for i in "${formats[@]}"; do
    mkdir processed_data
    mkdir docs/$i

    python process_data.py $i
    zip docs/$i/processed_data.zip processed_data/*
done

# Convert to pyodide
#
echo "Converting to pyodide..."
panel convert FormatAnalysis.py --to pyodide-worker --out docs/

# Add in the header information we need.
# This includes unpacking the zipped data and 
# adding the google analytics tag.
#
echo "Adding headers..."
python bundle.py

echo "Last updated: $(date)" > docs/data_version.txt