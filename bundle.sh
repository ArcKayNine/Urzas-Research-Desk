set -euo pipefail

# Run the processing and save in a zip.
#
# echo "Processing data..."
# formats=(
#     Standard
#     # Pioneer
#     # Modern
# )
# for i in "${formats[@]}"; do
#     mkdir docs/$i
#     echo "running process_data.py"
#     python process_data.py $i
#     echo "zipping"
#     zip docs/$i/processed_data.zip processed_data/*
# done

# Convert to pyodide
#
# panel convert exits 0 even when it fails, so build into a temp dir
# and check the output exists. Otherwise bundle.py would re-patch the
# stale (already patched) files in docs/.
#
echo "Converting to pyodide..."
build_dir=$(mktemp -d)
panel convert FormatAnalysis.py --skip-embed --to pyodide-worker --out "$build_dir"
if [ ! -f "$build_dir/FormatAnalysis.html" ] || [ ! -f "$build_dir/FormatAnalysis.js" ]; then
    echo "panel convert failed, aborting." >&2
    exit 1
fi
cp "$build_dir"/FormatAnalysis.* docs/
rm -rf "$build_dir"

# Add in the header information we need.
# This includes unpacking the zipped data and 
# adding the google analytics tag.
#
echo "Adding headers..."
python bundle.py

echo "Last updated: $(date)" > docs/data_version.txt