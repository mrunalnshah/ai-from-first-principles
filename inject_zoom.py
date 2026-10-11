"""Add click-to-zoom (_static/zoom.js) to every page of the built book.

Run it after the HTML build:
    jupyter book build --html
    python inject_zoom.py            # or: python inject_zoom.py path/to/_build/html

Safe to run twice: pages that already have it are skipped.
"""
from pathlib import Path
import sys

BUILD_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "_build/html")
SCRIPT = Path(__file__).parent / "_static" / "zoom.js"
MARKER = "data-myst-zoom"

if not BUILD_DIR.is_dir():
    sys.exit(f"No build found at {BUILD_DIR}. Run `jupyter book build --html` first.")

js = SCRIPT.read_text(encoding="utf-8")
if "</script" in js.lower():
    sys.exit("zoom.js must not contain '</script>'.")
tag = f"<script {MARKER}>{js}</script>"

added = 0
for page in BUILD_DIR.rglob("*.html"):
    html = page.read_text(encoding="utf-8")
    if MARKER in html:
        continue
    before, found, after = html.rpartition("</body>")
    if not found:
        continue
    page.write_text(before + tag + found + after, encoding="utf-8")
    added += 1

print(f"Click-to-zoom added to {added} page(s) in {BUILD_DIR}")
