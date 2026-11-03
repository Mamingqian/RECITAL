#!/usr/bin/env python3
"""Run PoET-2 in its isolated interpreter: generate CM pools, or score one request (the cm_poet sentinel)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("generate", "loglik"), default="generate")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--request", type=Path, help="loglik: JSON {context, sequences}")
    parser.add_argument("--response", type=Path, help="loglik: JSON destination")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--count", type=int, default=40960)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-count", type=int, default=64)
    parser.add_argument("--context-seed", type=int, default=2001)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--temperature", type=float, default=1.)
    args = parser.parse_args()
    if args.phase == "loglik":
        # The cm_poet sentinel: one prompt, one batch, one process.
        if args.source is None or args.checkpoint is None or args.request is None or args.response is None:
            parser.error("--source, --checkpoint, --request and --response are required for loglik")
        from tasks.protein.cm.poet2 import score_request

        result = score_request(args.request, args.response, source=args.source,
                               checkpoint=args.checkpoint, device=args.device,
                               batch_size=args.batch_size)
        print(json.dumps({k: v for k, v in result.items()
                          if k != "log_likelihood_per_token"}), flush=True)
        return
    if args.root is None or args.output is None or args.seed is None:
        parser.error("--root, --output and --seed are required for generation")
    if args.source is None or args.checkpoint is None:
        parser.error("--source and --checkpoint are required for generation")
    from tasks.protein.cm.poet2 import generate

    result = generate(args.root, args.output, source=args.source, checkpoint=args.checkpoint,
                      seed=args.seed, count=args.count, batch_size=args.batch_size,
                      context_count=args.context_count, context_seed=args.context_seed,
                      max_length=args.max_length, top_p=args.top_p, temperature=args.temperature,
                      device=args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
