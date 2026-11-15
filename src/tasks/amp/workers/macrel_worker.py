"""Persistent Macrel 1.6.1 scorer.

Run by `tasks.amp.oracles` in the Macrel virtual environment (`python -I macrel_worker.py`), not imported. It reads one JSON
request per line on stdin, `{"sequences": [...]}` of upper-case peptides, and writes one JSON reply per line on stdout,
`{"amp": [...], "hemolytic": [...]}` (the AMP and hemolysis probabilities of the bundled ONNX models) or `{"error": "..."}`.
Anything the libraries print goes to stderr.
"""
import json
import os
import sys

# Replies go through a private copy of the pipe; descriptor 1 is pointed at stderr so that what libraries and the
# child processes they start (perl, for ToxinPred3) print cannot corrupt the reply stream.
reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from macrel import AMP_predict  # noqa: E402
from macrel.macrel_features import compute_all, normalize_seq  # noqa: E402

COLUMNS = ["tinyAA", "smallAA", "aliphaticAA", "aromaticAA", "nonpolarAA", "polarAA", "chargedAA", "basicAA", "acidicAA",
           "charge", "pI", "aindex", "instaindex", "boman", "hydrophobicity", "hmoment", "SA.Group1.residue0",
           "SA.Group2.residue0", "SA.Group3.residue0", "HB.Group1.residue0", "HB.Group2.residue0", "HB.Group3.residue0"]
MODELS = AMP_predict.__file__.rsplit("/", 1)[0] + "/data/models/"


def score(sequences):
    kept = [normalize_seq(sequence) for sequence in sequences]
    table = pd.DataFrame([compute_all(sequence) for sequence in kept], columns=COLUMNS)
    table.insert(0, "group", "Unk")
    table.insert(0, "sequence", kept)
    result = AMP_predict.predict(MODELS + "AMP.onnx.gz", MODELS + "Hemo.onnx.gz", table, keep_negatives=True)
    return {"amp": np.asarray(result["AMP_probability"], dtype=float).tolist(),
            "hemolytic": np.asarray(result["Hemolytic_probability"], dtype=float).tolist()}


for line in sys.stdin:
    try:
        sequences = json.loads(line)["sequences"]
        answer = score(sequences) if sequences else {"amp": [], "hemolytic": []}
    except Exception as error:  # reported to the caller, which fails the scoring step
        answer = {"error": f"{type(error).__name__}: {error}"}
    reply.write(json.dumps(answer) + "\n")
    reply.flush()
