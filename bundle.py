from pathlib import Path

# Grab the format data passed to the worker.
# Then fetch and unzip the right data.
#
JS_IN_FUNC = '''console.log("Environment loaded!");
  self.postMessage({type: 'status', msg: 'Reading Data'});
  console.log('Starting application with format:', appFormat);
    
  // Fetch the format data we want
  let zipResponse = await fetch(`${appFormat}.zip`);
  let zipBinary = await zipResponse.arrayBuffer();
  self.pyodide.unpackArchive(zipBinary, "zip");
  console.log("Data loaded!");'''

# Set up global stuff to handle passing
# format data around.
#
JS_GLOBAL = '''full/pyodide.js");

// Global variable to store the format
let appFormat = 'Standard'; // Default value

// Listen for messages from the main thread
self.addEventListener('message', function(e) {
    if (e.data.type === 'init' && e.data.params) {
        // Store the format parameter
        appFormat = e.data.params.format || 'Standard';
        console.log('Received format parameter:', appFormat);
    }
    // Handle other messages...
});'''

js_file = Path('docs/FormatAnalysis.js')
js_file.write_text(
    js_file.read_text().replace(
        'console.log("Environment loaded!");', 
        JS_IN_FUNC
    ).replace(
        'full/pyodide.js");',
        JS_GLOBAL
    ))

# Read the format data from the url parameters,
# then pass them to the pyodide worker.
#
HTML_FORMAT_INFO = '''const urlParams = new URLSearchParams(window.location.search);
      const format = urlParams.get('format') || 'Standard';

      const pyodideWorker = new Worker("./FormatAnalysis.js");
      pyodideWorker.postMessage({
        type: 'init',
        params: {
            format: format
        }
      });'''

# Set up google analytics.
# Also set up mtgify stuff (for now).
#
HTML_HEADER = '''</head>
  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-95C5QTVN8T"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-95C5QTVN8T');
  </script>
  '''

html_file = Path('docs/FormatAnalysis.html')
html_file.write_text(
    html_file.read_text().replace(
        '</head>', 
        HTML_HEADER
    ).replace(
        'const pyodideWorker = new Worker("./FormatAnalysis.js");',
        HTML_FORMAT_INFO
    ))