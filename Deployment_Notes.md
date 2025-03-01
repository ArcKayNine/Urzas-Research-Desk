# Pyodide conversion

Run the following:
```
panel convert index.py --to pyodide-worker --out docs/
```

Then we need to make sure pyodide can see the files we want it to, so we'll fetch them and unzip them.
processed_data.zip should be in docs/.
The following lines go in index.js just under `console.log("Packages loaded!");`

```
  self.postMessage({type: 'status', msg: 'Reading Data'});
  let zipResponse = await fetch("processed_data.zip");
  let zipBinary = await zipResponse.arrayBuffer();
  self.pyodide.unpackArchive(zipBinary, "zip");
```

