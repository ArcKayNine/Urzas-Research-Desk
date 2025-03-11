console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Reading Data'});
  let zipResponse = await fetch("processed_data.zip");
  let zipBinary = await zipResponse.arrayBuffer();
  self.pyodide.unpackArchive(zipBinary, "zip");
  console.log("Data loaded!");