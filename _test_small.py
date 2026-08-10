import sys
sys.path.insert(0, '/home/junwin/src/repos/image_description')
from src.image_description.cli.wpcom_cli import _convert_markdown_body

with open('/home/junwin/src/repos/junwin.github.io/_posts/2026-07-09-A67A0191.md', 'r') as f:
    content = f.read()

parts = content.split('---', 2)
body = parts[2].strip()
html = _convert_markdown_body(body)

print('=== LAST 800 CHARS ===')
print(html[-800:])
