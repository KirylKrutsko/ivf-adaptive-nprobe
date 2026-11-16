"""Embed presentation/data.json into presentation/adaptive-nprobe.html (between the data markers)."""
import re
from pathlib import Path

here = Path(__file__).parent
html = (here / "adaptive-nprobe.html").read_text()
data = (here / "data.json").read_text()
new = re.sub(r'(<script id="data" type="application/json">)(.*?)(</script>)',
             lambda m: m.group(1) + data + m.group(3), html, count=1, flags=re.S)
(here / "adaptive-nprobe.html").write_text(new)
print("embedded", len(data), "bytes of data")
