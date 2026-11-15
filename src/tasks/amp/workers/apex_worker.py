"""Persistent APEX scorer (40 pickled modules: 8 configurations x 5 members), on CPU.

Run by `tasks.amp.oracles` (`python -I apex_worker.py APEX_DIR`), not imported. APEX_DIR is the clone of the APEX repository
(`AMP_DL_model_twohead.py`, `utils.py`, `aaindex1.csv`, `best_key_list`, `trained_models/`); both modules were read before use
and the 40 files import only torch.nn modules and `AMP_DL_model_twohead` classes. One JSON request per line,
`{"sequences": [...]}`; one reply per line, `{"mic": [[E. coli ATCC11775, S. aureus ATCC12600, P. aeruginosa PAO1], ...]}`
in micromolar (the mean over members of `10 ** (6 - output)`, as APEX's `predict.py` computes it) or `{"error": "..."}`.
"""
import ast
import json
import os
import sys

# Replies go through a private copy of the pipe; descriptor 1 is pointed at stderr so that what libraries and the
# child processes they start (perl, for ToxinPred3) print cannot corrupt the reply stream.
reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

APEX_DIR = sys.argv[1]
sys.path.insert(0, APEX_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils import AAindex, make_vocab, onehot_encoding  # noqa: E402

STRAINS = ("E. coli ATCC11775", "S. aureus ATCC12600", "P. aeruginosa PAO1")
MAX_LEN = 52
REPEATS = 5

torch.set_num_threads(int(os.environ.get("APEX_THREADS", "4")))
columns = next(ast.literal_eval(node.value) for node in ast.parse(open(os.path.join(APEX_DIR, "predict.py")).read()).body
               if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "col")
indices = [columns.index(strain) for strain in STRAINS]
word2idx, _ = make_vocab()
AAindex(os.path.join(APEX_DIR, "aaindex1.csv"), word2idx)
members = []
for key in [line.strip() for line in open(os.path.join(APEX_DIR, "best_key_list")) if line.strip()]:
    for repeat in range(REPEATS):
        # whole pickled modules: weights_only=False is required, see the module docstring
        members.append(torch.load(os.path.join(APEX_DIR, "trained_models", f"trained_all_model_{key}_ensemble_{repeat}"),
                                  map_location="cpu", weights_only=False).eval())


def score(sequences):
    encoded, _, _ = onehot_encoding(np.array(sequences), MAX_LEN, word2idx)
    batch = torch.LongTensor(encoded)
    total = 0.0
    with torch.no_grad():
        for member in members:
            total = total + 10 ** (6 - member(batch).numpy())
    return {"mic": (total / len(members))[:, indices].tolist()}


for line in sys.stdin:
    try:
        sequences = json.loads(line)["sequences"]
        answer = score(sequences) if sequences else {"mic": []}
    except Exception as error:
        answer = {"error": f"{type(error).__name__}: {error}"}
    reply.write(json.dumps(answer) + "\n")
    reply.flush()
