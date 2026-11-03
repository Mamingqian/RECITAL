#!/usr/bin/env python3
"""Prepare and qualify the cross-dataset CM function models."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "esm2-650m", "prot-t5", "summary"), required=True)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--inference-smoke", action="store_true")
    args = parser.parse_args()
    import torch
    from tasks.protein.cm.qualification import prepare_qualification, qualify_encoder, summarize_qualification
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    if args.stage == "prepare":
        result = prepare_qualification(args.root, args.pilot_root, seed=args.seed)
    elif args.stage == "summary":
        result = summarize_qualification(args.root)
        result = {"complete": result["complete"], "models": list(result["models"])}
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Submit encoder extraction on a GPU node")
        if args.inference_smoke:
            import numpy as np
            from tasks.protein.cm.predictors import ESMFeatures, ProtT5Features
            from tasks.protein.cm.qualification import write_json
            with np.load(args.root / "oracles" / f"{args.stage}_features.npz", allow_pickle=False) as cache:
                sequences = cache["sequences"][:16].tolist()
                expected = cache["features"][:16]
            cls = ProtT5Features if args.stage == "prot-t5" else ESMFeatures
            encoder = cls(args.root / "assets" / args.stage, device=args.device)
            actual = encoder(sequences)
            error = float(np.sqrt(np.mean((actual - expected) ** 2)))
            if not np.isfinite(actual).all() or error > .02:
                raise ValueError(f"Frozen encoder inference mismatch: RMS={error}")
            result = {"model": args.stage, "device": args.device, "n": len(sequences),
                      "cache_vs_live_feature_rms_error": error}
            write_json(args.root / "oracles" / f"{args.stage}_inference_smoke.json", result)
            print(json.dumps(result), flush=True)
            return
        result = qualify_encoder(args.root, args.stage, device=args.device, seed=args.seed,
                                 pilot_root=args.pilot_root)
        result = {"feature_model": args.stage, "elapsed_seconds": result["elapsed_seconds"]}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
