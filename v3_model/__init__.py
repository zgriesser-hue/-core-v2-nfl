"""
v3_model
========

Clean, tested extraction of the frozen/validated V3.0-V3.4 research
from the Colab notebooks (V3.ipynb) into production-ready modules.

This package intentionally does NOT try to reproduce Core V2's
internal math. It implements V3's OWN validated design (V3.6, the
"Pure V3" integration frozen in the notebook's V3.6B/V3.6E cells),
using V2 only as an external sanity-check baseline once live 2026
results start coming in.

Pipeline (mirrors notebook stages):

    rating.py     -> V3.0  prior-anchored latent offense/defense states
    spread.py     -> V3.0  HFA + EPA-to-points margin calibration
    personnel.py  -> V3.2  major defensive-attrition adjustment
    market.py     -> V3.3  lagged (never-current-game) market signal
    neutral.py    -> V3.4  true-neutral-site HFA removal
    pipeline.py   -> V3.6  combines all of the above: pred_v36

See README.md for what is carried over exactly from the notebooks,
what still needs a one-time production calibration pass on your
full history, and how to run the test suite.
"""

from .pipeline import predict_v36  # noqa: F401
