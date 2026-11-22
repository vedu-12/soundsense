"""Block 0 checks: folder structure, docs and core imports."""
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parents[1]


def test_folders_exist():
    for d in ["docs", "tests", "data", "src", "notebooks", "models", "results/figures", "app/pages"]:
        assert (ROOT / d).is_dir(), d


def test_reference_docs_in_docs():
    names = [p.name for p in (ROOT / "docs").iterdir()]
    assert any("Proposal" in n for n in names)
    assert any("Team Briefs" in n for n in names)


def test_core_files():
    for f in ["CLAUDE.md", "README.md", "requirements.txt", ".gitignore", "docs/BRIEF_CHECKLIST.md"]:
        assert (ROOT / f).is_file(), f


def test_gitignore_keeps_outputs():
    gi = (ROOT / ".gitignore").read_text()
    assert "data/raw/" in gi
    for keep in ["data/processed", "models", "results"]:
        assert keep not in gi


def test_imports():
    for m in ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "plotly", "streamlit", "joblib"]:
        importlib.import_module(m)
