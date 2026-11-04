# models/

Intentionally empty. The original project does not train or save a deployable model:
the classifiers are only evaluated with cross-validation, and the recommendation step is a
nearest-centroid lookup (see `docs/DISCREPANCIES.md` B4). If you add model persistence
(e.g. `joblib.dump`), put the artefacts here and document them.
