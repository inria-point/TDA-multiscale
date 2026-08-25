"""Do two perturbations add up?

The models we cannot reach need the coarse scale up and the fine scale down at
once. We have one perturbation for each half separately -- the literary
rewrite raises the coarse scale by 12%, a quarter-strength echo lowers the
fine one by 18% -- and nothing that does both.

Composing them tests two things at once. Whether the pair lands where the
generators are, and, more generally, whether the map is additive: if the
composition sits at the sum of the two displacements, perturbations can be
combined by design, and if it does not, every combination has to be measured.

The order matters, so both are run: the echo is mechanical and blind to
meaning, and applying it before or after the rewrite is not the same thing.
Here only echo-after-rewrite is possible, since the rewrite came from a model
we are no longer calling.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from perturb import apply

BASE = os.path.join(HERE, "..")

if __name__ == "__main__":
    src = json.load(open(os.path.join(BASE, "results",
                                      "coling_collapse_probes.json")))
    lit = src["style_literary_apiyi"]
    out = {}
    for name, pert in [("literary_echo25", "echo_p25"),
                       ("literary_echo50", "echo_p50"),
                       ("literary_ctxlock", None)]:
        if pert is None:
            continue
        out[name] = {k: apply(pert, v, seed=i)
                     for i, (k, v) in enumerate(lit.items())}
    path = os.path.join(BASE, "results", "coling_composed.json")
    json.dump(out, open(path, "w"), ensure_ascii=False)
    print({k: len(v) for k, v in out.items()}, "->", path)
