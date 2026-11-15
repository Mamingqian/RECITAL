#!/usr/bin/env python
"""Run `module_contrast.py`, `baseline_reading.py` or `amp_final_value_contrast.py` on a scripted search, whose resolved model is `arm-random`.

    scripted_same_model.py module_contrast amp --arm A.csv --comparator C.csv --declared-tag TAG --declared-path PATH
    scripted_same_model.py baseline_reading amp --arm A.csv --comparator C.csv --declared-tag TAG --declared-path PATH
    scripted_same_model.py final_value --arm A.csv --comparator C.csv --n-arm 6 --n-comparator 6

`module_contrast.compatible` requires the arm and its comparator to share the resolved model; a scripted search makes no model call, so its
model (`arm-random`) differs from the comparators' `gpt-6-luna`. This wrapper replaces `compatible` by a function that skips the same-model
check when, and only when, every cell of the `arm` group has the resolved model `arm-random`; every other check (total updates, protocol
condition, endpoint criterion, no cell in two groups, no equal first-event time) and every threshold, function and output of the three
scripts are the audited ones, which are not edited.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import baseline_reading as br  # noqa: E402
import module_contrast as mc  # noqa: E402  (the module object `baseline_reading` and `module_contrast` run on)
from analysis import module_contrast as mc_package  # noqa: E402  (the one `amp_final_value_contrast` imports)

SCRIPTED_MODEL = "arm-random"
MODULES = (mc, mc_package)
_ORIGINALS = {id(module): module.compatible for module in MODULES}


def guarded(original):
    def compatible(groups, same_model=()):
        arm = groups.get("arm")
        scripted = arm is not None and {str(v) for v in arm["models"]} == {SCRIPTED_MODEL}
        original(groups, () if scripted else same_model)
    return compatible


def install():
    for module in MODULES:
        module.compatible = guarded(_ORIGINALS[id(module)])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("module_contrast", "baseline_reading", "final_value"):
        raise SystemExit(__doc__)
    install()
    script, rest = argv[0], argv[1:]
    if script == "module_contrast":
        return mc.main(rest)
    if script == "baseline_reading":
        return br.main(rest)
    import amp_final_value_contrast as fv
    sys.argv = ["amp_final_value_contrast.py", *rest]
    fv.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
