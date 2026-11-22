"""Helpers to check notebooks: saved with outputs, no errors; optional fresh execution."""
import os
import subprocess
import sys
from pathlib import Path

import nbformat

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def check_notebook(name):
    path = NB_DIR / name
    assert path.exists(), f"{name} missing"
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "no code cells"
    for c in code:
        assert c.execution_count is not None, f"{name}: unexecuted cell"
        for o in c.get("outputs", []):
            assert o.output_type != "error", f"{name}: error output {o.get('ename')}"
    md = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for heading in ["What this part does", "Concepts in plain English", "Results summary", "Questions you may be asked"]:
        assert heading in md, f"{name}: missing section '{heading}'"
    if os.environ.get("RUN_NOTEBOOKS") == "1":       # slow: re-execute from scratch
        jup = Path(sys.executable).parent / "jupyter"
        r = subprocess.run([str(jup), "nbconvert", "--to", "notebook", "--execute", "--stdout",
                            "--ExecutePreprocessor.timeout=1800", str(path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
