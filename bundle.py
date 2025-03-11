from pathlib import Path

js_header = Path('js_header.js').read_text()
js_file = Path('docs/index.js')
js_file.write_text(js_file.read_text().replace('console.log("Packages loaded!");', js_header))

html_header = Path('html_header.html').read_text()
html_file = Path('docs/index.html')
html_file.write_text(html_file.read_text().replace('</head>', html_header))