"""Persistent ToxinPred3 1.4 scorer.

Run by `tasks.amp.oracles` in the ToxinPred3 virtual environment (`python -I toxinpred3_worker.py SCRATCH`), not imported.
One JSON request per line on stdin, `{"sequences": [...]}`; one reply per line on stdout, `{"hybrid": [...], "ml": [...]}`
(the `Hybrid Score` and `ML Score` columns of `toxinpred3 -m 2 -t 0.38`) or `{"error": "..."}`. The package writes
`Sequence_1`, `seq.aac`, `seq.dpc` and `merci_*.txt` into its working directory and calls `perl`, so the worker works in a
temporary directory of its own under SCRATCH.
"""
import contextlib
import json
import os
import sys
import tempfile

# Replies go through a private copy of the pipe; descriptor 1 is pointed at stderr so that what libraries and the
# child processes they start (perl, for ToxinPred3) print cannot corrupt the reply stream.
reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

import pandas as pd  # noqa: E402
from toxinpred3.python_scripts import toxinpred3 as package  # noqa: E402

os.makedirs(sys.argv[1], exist_ok=True)
WORK = tempfile.mkdtemp(prefix="toxinpred3_", dir=sys.argv[1])
os.chdir(WORK)


# MERCI_Processor_n (toxinpred3.py:192-197) gives every row of a call the -0.5 offset when no row of the call matches a non-toxic motif,
# so a peptide's score would depend on the other rows (a lone peptide with no motif scores 0.5 below the same peptide in a call that has a
# motif-carrying row). Every call therefore carries HIT, a peptide that matches a non-toxic motif, and drops its row; the same padding
# keeps a call from ever being one row, which the package cannot score (numpy.AxisError). NO_HIT matches none.
HIT = "LRIRLIRRFIRRIIRQLIRRVR"
NO_HIT = "KAPLLNCSFGLLPS"


def table(sequences):
    with open("in.fasta", "w") as handle:
        handle.write("".join(f">s{i}\n{sequence}\n" for i, sequence in enumerate(sequences)))
    sys.argv = ["toxinpred3", "-i", "in.fasta", "-o", "out.csv", "-t", "0.38", "-m", "2", "-d", "2"]
    with contextlib.redirect_stdout(sys.stderr):
        package.main()
    return pd.read_csv("out.csv").set_index("Subject").loc[[f"s{i}" for i in range(len(sequences))]]


def score(sequences):
    rows = table([*sequences, HIT]).iloc[:len(sequences)]
    return {"hybrid": rows["Hybrid Score"].astype(float).tolist(), "ml": rows["ML Score"].astype(float).tolist()}


# The padding works only while HIT matches a motif and NO_HIT does not: with HIT in the call, NO_HIT keeps the offset 0.
offsets = table([HIT, NO_HIT])["MERCI Score Neg"].tolist()
if offsets != [-0.5, 0.0]:
    raise SystemExit(f"ToxinPred3 self-check failed: the non-toxic-motif offsets of the filler pair are {offsets}, not [-0.5, 0.0]")


for line in sys.stdin:
    try:
        sequences = json.loads(line)["sequences"]
        answer = score(sequences) if sequences else {"hybrid": [], "ml": []}
    except BaseException as error:  # SystemExit from the package's argparse included
        answer = {"error": f"{type(error).__name__}: {error}"}
    reply.write(json.dumps(answer) + "\n")
    reply.flush()
