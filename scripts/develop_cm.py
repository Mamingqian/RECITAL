#!/usr/bin/env python3
"""Prepare the CM prior's common score scale (`development/prior_scale.json`)."""
import argparse
import json
from pathlib import Path

import torch

from tasks.protein.cm.development import prepare_prior_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--prepare-scale", action="store_true", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("the CM prior scale needs a GPU allocation")
    result = prepare_prior_scale(args.root, args.pilot_root, device="cuda")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
