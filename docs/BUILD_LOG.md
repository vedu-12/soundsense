# SoundSense Build Log

## Block 0: Setup  (commit see git log: "Block 0: setup")
- Built: folder structure, `requirements.txt`, `.gitignore`, `README.md` stub, `docs/BRIEF_CHECKLIST.md`, `tests/test_block0.py`, `.venv` (Python 3.14.6), Playwright Chromium, GitHub CLI 2.105.0 (installed via winget).
- Checks run: `pytest tests/test_block0.py` (structure, docs, imports) passed. Library versions: pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, plotly 6.9.0, streamlit 1.67.0, joblib 1.6.0, scipy 1.18.1.
- Key numbers: n/a
- Decisions I made: Python 3.14 is the only interpreter on the machine; all libraries installed fine on it. Requirements pin major-version ranges. Repo private (default).
- Differences between the documents and CLAUDE.md (followed CLAUDE.md, recorded here):
  1. Briefs list inputs without `time_signature`; proposal and CLAUDE.md include it (one-hot, or drop if almost constant). Decided in Block 3.
  2. P4 brief says "plot **test** accuracy vs k"; CLAUDE.md says CV on train (never tune on test). The notebook plots **cross-validated** accuracy vs k and uses the test set once for the chosen k.
  3. P6 brief says tune hidden neurons and **learning rate**; CLAUDE.md says hidden size and **alpha**. The notebook tunes all three with a small CV grid.
  4. P1 brief says "duplicate track_ids are removed"; CLAUDE.md additionally removes tracks in more than one chosen genre and same name+artist duplicates. Both are done.
  5. Hit threshold "≥ 70" in briefs/proposal may change by the CLAUDE.md fallback rule if the hit rate is under 5% (decided in Block 2).
