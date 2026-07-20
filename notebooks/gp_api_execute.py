"""Execute gp_api.ipynb in place, embedding outputs, and refresh the HTML preview.

The preview is written on every full run, because keeping it in step by hand does
not work: it silently drifted three times in one session, each time reflecting an
older notebook than the committed one.

``--fast`` runs it as a smoke test instead: inference settings are shrunk to a
few draws/iterations so the whole notebook runs in well under a minute, and the
result is thrown away rather than written back. Use it to check the notebook
still runs end to end after a code change; use a full run to produce the
deliverable, since the fast one's numbers and figures are noise.
"""

import pathlib
import re
import subprocess
import sys

import nbformat

from nbclient import NotebookClient

FAST = "--fast" in sys.argv

# (pattern, replacement) applied to every code cell in --fast mode.
FAST_SUBS = [
    (r"draws=\d+", "draws=25"),
    (r"tune=\d+", "tune=25"),
    (r"trainer\.fit\(\d+\)", "trainer.fit(50)"),
    (r"n_particles=\d+", "n_particles=4"),
    (r"range\(0, post\.sizes\[\"sample\"\], \d+\)", 'range(0, post.sizes["sample"], 20)'),
]

here = pathlib.Path(__file__).parent
path = here / "gp_api.ipynb"

nb = nbformat.read(path, as_version=4)
_, nb = nbformat.validator.normalize(nb)  # cell ids; missing ones are a future hard error

if FAST:
    for cell in nb.cells:
        if cell.cell_type == "code":
            for pattern, replacement in FAST_SUBS:
                cell.source = re.sub(pattern, replacement, cell.source)

client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    resources={"metadata": {"path": str(here)}},
    allow_errors=False,
)

try:
    client.execute()
except Exception as exc:
    if not FAST:
        nbformat.write(nb, path)
    print(f"EXECUTION FAILED: {type(exc).__name__}: {str(exc)[:2000]}")
    sys.exit(1)

if FAST:
    print(f"OK (fast): {len(nb.cells)} cells ran end to end; outputs discarded")
    sys.exit(0)

nbformat.write(nb, path)

subprocess.run(
    [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--output",
        "gp_api_preview.html",
        str(path),
    ],
    capture_output=True,
    text=True,
    cwd=str(here),
)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
n_img = sum(
    1
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.outputs
    if "image/png" in getattr(o, "get", lambda *_: {})("data", {})
)
print(
    f"OK: {len(nb.cells)} cells, {n_code} code cells, {n_out} produced output, "
    f"{n_img} figures; preview refreshed"
)
