# What comes after five songs?

Predict the immediate sixth track from five consecutive tracks in an ordered Spotify
playlist. The current deliverable is **`assignment_2.ipynb`**, with all five assignment
sections: task, data, modeling, evaluation, and related work/findings.

The central comparison adds audio descriptors to a logistic ranker that already uses
history and artist compatibility. Baselines include random ranking in expectation,
popularity, last artist, item cosine, Markov transitions, and audio similarity. All
evaluation queries rank the full catalogue; previously played songs remain eligible.

## Run

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe continuation_experiment.py
./.venv/Scripts/python.exe verify_continuation.py
```

To rebuild and execute the documented notebook:

```powershell
./.venv/Scripts/python.exe build_continuation_notebook.py
```

Run from the repository root. The notebook also contains the implementation and can
be run top to bottom without importing the experiment module. Dataset discovery checks
`SONGS620_PATH`, `songs620.csv` beside the notebook, `data/songs620.csv`, and a Desktop
copy, in that order. A set but invalid environment override produces an explicit error.

## Protocol

- Seed-42 permutation of sorted playlist IDs: 800 train, 100 validation, 100 test.
- All training five-track windows predict their immediate successor; primary validation
  and test use one opening query per eligible playlist (98 validation, 100 test).
- Four playlist folds cross-fit relational training features to exclude the example's
  own playlist. Final support statistics use only the 800 training playlists.
- Same 1:10 uniform pseudo-negative samples and playlist weights for all learned models.
- C selected from a fixed grid using validation HR@10, then frozen before test evaluation.
- Primary paired audio comparison: 10,000-playlist bootstrap interval and exact paired test.
- Frozen-model order and later-position sensitivity checks are secondary diagnostics.

The dataset has no timestamps or listener IDs. The known static catalogue and
curator ordering define the prediction setting; satisfaction and chronological
generalization are not observed. Exact provenance of the enriched local CSV is unverified.

## Outputs

Current outputs are in **`output/next_song/`**: manifests, query definitions, validation
grid, selected configurations, coefficients, full-catalogue test ranks, top-ten lists,
paired audio results, subgroup and sensitivity metrics, figures, and environment details.
`requirements-lock.txt` records the local environment; `requirements.txt` lists the
packages needed for the experiment and notebook.

No new HTML export is generated. Existing `workbook.html`, `experiment.py`,
`build_notebook.py`, `verify_experiment.py`, and top-level `output/` results belong to
the earlier **random missing-track** task, not this immediate-successor experiment.
Its notebook is preserved at `output/legacy_missing_track_assignment.ipynb`.
