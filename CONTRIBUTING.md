# Contributing

Thanks for helping improve this project. It accompanies a published paper, so the
guiding rule is: **do not silently change published methodology or results.**

## Reporting issues

Use the issue templates. Please include: what you ran (exact command and config),
your Python and package versions (`pip freeze | grep -E "numpy|pandas|scikit|folium"`),
and the full error. **Never attach the proprietary AbsoluIT data or any real vehicle
location data**; use `make-sample` output instead.

## Development setup

```bash
git clone <your fork>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python -m pytest
```

## Pull requests

1. Open an issue first for larger changes.
2. Keep changes focused; add or update tests in `tests/`.
3. `python -m pytest` must pass.
4. If behaviour differs from the original notebooks or the paper, document it in
   `docs/DISCREPANCIES.md` and `CHANGELOG.md` instead of hiding it.
5. Mark anything new that is not part of the paper as *new/proposed* in the docs.
6. Do not claim a result or a test you have not actually run.
7. Docstrings for new public functions; follow the style of the existing modules.

## Good first contributions

Time-aware centroids, scaling/projection options, elbow/silhouette selection of *k*,
a persisted model, extra visualisations, documentation fixes.

## Reproducibility contributions

If you have authorized access to the dataset, reporting the values of
`cluster-metrics` and your package versions against the paper's table is very valuable
(open an issue titled "Reproduction report").
