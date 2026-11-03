#!/usr/bin/env python3
"""Prepare CM data, fit the prerequisite models and train the SFT policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "oracles", "sft"), required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        from tasks.protein.cm.data import prepare_data, download_models
        result = prepare_data(args.root, seed=args.seed)
        download_models(args.root)
    else:
        import torch
        torch.set_num_threads(8)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CM stage={args.stage}, device={device}, torch={torch.__version__}, root={args.root}", flush=True)
        if args.stage == "oracles":
            from tasks.protein.cm.predictors import train_predictors
            result = train_predictors(args.root, device=device, seed=args.seed)
        else:
            from tasks.protein.cm.runner import run_stage
            result = run_stage(args, device=device)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
