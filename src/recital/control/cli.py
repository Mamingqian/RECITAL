"""Command line of the control layer: preflight, run one cell, run a campaign, read status.

    python -m recital.control.cli preflight --config C --output DIR
    python -m recital.control.cli run --config C --output DIR [--arm A --seed N --scheduler slurm]
    python -m recital.control.cli campaign --config C --output-root ROOT --arms A B C --seeds 0 1 2
    python -m recital.control.cli fixed-recipe --config C --output-root ROOT --steps 4000 8192 --seeds 0 1 2
    python -m recital.control.cli status DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from .campaign import Cell, JobGate, run_campaign
from .contract import ControlContract, endpoint_host
from .errors import ControlError
from .export import export
from .launch import ARM, Prepared, prepare, require_arm, reward_role_policy
from .runtime_build import build_runtime, preflight
from .scheduler import LocalScheduler, Scheduler
from .state import fold
from .store import read_events
from .events import from_record
from .transport import ResponsesTransport, read_key_file

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = (ARM,)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ControlError("the run config must be a mapping")
    for key, target in (("seed", "seed"), ("total_updates", "total_updates"), ("arm", "protocol_condition")):
        value = getattr(args, key, None)
        if value is not None:
            config[target] = value
    if getattr(args, "max_concurrent_jobs", None) is not None:
        config["scheduler"]["max_concurrent_jobs"] = args.max_concurrent_jobs
    return config


def make_scheduler(kind: str, run_dir: Path) -> Scheduler:
    if kind == "local":
        return LocalScheduler([run_dir / "jobs"])
    if kind == "slurm":
        from .slurm import SlurmScheduler
        return SlurmScheduler()
    raise ControlError(f"unknown scheduler {kind!r}")


def make_transport(config: dict[str, Any]) -> ResponsesTransport:
    llm = ControlContract.from_config(config).llm
    return ResponsesTransport(model=llm.model, max_output_tokens=llm.max_output_tokens, base_url=llm.base_url,
                              api_key_file=llm.api_key_file, schema_dialect=llm.schema_dialect)


def make_components(parts: dict[str, Any]):
    from .coder import build_runner
    return build_runner(parts["settings"].control, backend=parts["backend"], synthesis=parts["synthesis"],
                        transport=parts["transport"])


def cmd_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config, args)
    control = ControlContract.from_config(config)
    if control.llm.api_key_file is not None:
        read_key_file(control.llm.api_key_file)  # a cell that names a key file cannot start without it
    runtime = build_runtime(config, repo_root=args.repo_root, output=args.output, device="cpu",
                            audit_oracle_factories={})
    declaration = preflight(runtime, device=args.device)
    task = runtime["contract"]
    require_arm(task)
    control.validate_against(task.visible_sources, list(task.conditions), int(config["delivery"]["count"]))
    reward_role_policy(runtime)  # raises when this arm cannot see a source the task's policy requires enabled
    print(json.dumps({"declaration": declaration, "control_contract_hash": control.contract_hash(),
                      "llm": {"model": control.llm.model, "endpoint": endpoint_host(control.llm.base_url)}},
                     indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


def _install_stop_flag() -> list[bool]:
    flag = [False]

    def handler(signum, frame):  # noqa: ARG001 - signature fixed by `signal`
        flag[0] = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return flag


def audit_device() -> str:
    """The controller's own device for the terminal blind evaluation, which may run on a node with no GPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _finish(prepared: Prepared, args: argparse.Namespace) -> dict[str, Any]:
    state = prepared.engine.state()
    result: dict[str, Any] = {"status": state.finished_status, "output": str(prepared.settings.run_dir)}
    if state.finished_status == "frozen" and not args.no_audit:
        from .terminal_audit import run_terminal_audit
        result["terminal_audit"] = run_terminal_audit(
            yaml.safe_load(prepared.settings.config_path.read_text()), prepared.settings.run_dir,
            repo_root=prepared.settings.repo_root, device=audit_device())["status"]
    return result


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args)
    run_dir = args.output.expanduser().resolve()
    stop = _install_stop_flag()
    prepared = prepare(config, run_dir=run_dir, repo_root=args.repo_root, scheduler=make_scheduler(args.scheduler, run_dir),
                       transport=make_transport(config), device=args.device, components=make_components)
    try:
        cell = Cell(run_dir.name, prepared.engine)
        run_campaign([cell], poll_seconds=prepared.settings.control.scheduler.poll_seconds,
                     should_stop=lambda: stop[0])
        print(json.dumps(_finish(prepared, args) if cell.done else {"status": "interrupted"}, sort_keys=True))
    finally:
        prepared.close()
    return 0


def cell_dir(root: Path, config: dict[str, Any], arm: str, seed: int) -> Path:
    return root / f"{config.get('task_family')}_{arm}_seed{seed}"


def cmd_campaign(args: argparse.Namespace) -> int:
    base = load_config(args.config, args)
    cap = base["scheduler"]["max_concurrent_jobs"]
    gate, prepared_cells, cells = JobGate(cap), [], []
    stop = _install_stop_flag()
    try:
        # The job gate serves cells in this order, so each seed's arms are adjacent and advance together.
        for seed in args.seeds:
            for arm in args.arms:
                config = {**base, "protocol_condition": arm, "seed": seed}
                run_dir = cell_dir(args.output_root.expanduser().resolve(), config, arm, seed)
                prepared = prepare(config, run_dir=run_dir, repo_root=args.repo_root,
                                   scheduler=make_scheduler(args.scheduler, run_dir), transport=make_transport(config),
                                   device=args.device, components=make_components, may_submit=gate.allow)
                prepared_cells.append(prepared)
                cells.append(Cell(run_dir.name, prepared.engine))
        gate.bind(cells)
        states = run_campaign(cells, poll_seconds=base["scheduler"]["poll_seconds"], should_stop=lambda: stop[0])
        summary = {name: state.finished_status for name, state in states.items()}
        if not stop[0] and not args.no_audit:
            for cell, prepared in zip(cells, prepared_cells):
                if states[cell.name].finished_status == "frozen":
                    summary[cell.name] = {"frozen": _finish(prepared, args).get("terminal_audit")}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for prepared in prepared_cells:
            prepared.close()
    return 0


def cmd_drill_coder(args: argparse.Namespace) -> int:
    from .drill import run_coder_drill

    config = load_config(args.config, args)
    run_dir = args.output.expanduser().resolve()
    result = run_coder_drill(
        config, run_dir=run_dir, repo_root=args.repo_root, scheduler=make_scheduler(args.scheduler, run_dir),
        coder_transport=make_transport(config), device=args.device,
        poll_seconds=config["scheduler"]["poll_seconds"], oracles=tuple(args.drill_oracles))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "frozen" and all(
        item["status"] == "published" for item in result["components"].values()) else 1


def cmd_fixed_recipe(args: argparse.Namespace) -> int:
    """No-search control cells: the declared initial recipe trained once for each of `--steps`, per seed."""
    from .fixed_recipe import prepare_cell

    base = load_config(args.config, args)
    gate, prepared_cells, cells = JobGate(base["scheduler"]["max_concurrent_jobs"]), [], []
    stop = _install_stop_flag()
    try:
        for seed in args.seeds:
            for steps in args.steps:
                config = {**base, "protocol_condition": "no_self_validation", "seed": seed}
                run_dir = args.output_root.expanduser().resolve() / f"{config.get('task_family')}_fixedrecipe_steps{steps}_seed{seed}"
                prepared = prepare_cell(config, steps=steps, run_dir=run_dir, repo_root=args.repo_root,
                                        scheduler=make_scheduler(args.scheduler, run_dir), device=args.device,
                                        may_submit=gate.allow)
                prepared_cells.append(prepared)
                cells.append(Cell(run_dir.name, prepared.engine))
        gate.bind(cells)
        states = run_campaign(cells, poll_seconds=base["scheduler"]["poll_seconds"], should_stop=lambda: stop[0])
        summary = {name: state.finished_status for name, state in states.items()}
        if not stop[0] and not args.no_audit:
            for cell, prepared in zip(cells, prepared_cells):
                if states[cell.name].finished_status == "frozen":
                    summary[cell.name] = {"frozen": _finish(prepared, args).get("terminal_audit")}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for prepared in prepared_cells:
            prepared.close()
    return 0


def status_line(run_dir: Path) -> dict[str, Any]:
    events = read_events(run_dir / "events" / "log.jsonl")
    if not events:
        return {"run": run_dir.name, "status": "not_started"}
    state = fold(from_record(event) for event in events)
    counts: dict[str, int] = {}
    for candidate in state.candidates.values():
        counts[candidate.status] = counts.get(candidate.status, 0) + 1
    usage = state.usage
    attention = run_dir / "needs_attention.json"
    return {
        "run": run_dir.name, "status": state.finished_status or "running", "rounds_done": state.rounds_done,
        "open_round": state.open_round.index if state.open_round else None, "candidates": counts,
        "baseline": state.baseline.status, "committed_updates": state.committed_updates,
        "decisions": len(state.decisions),
        "invalid_decisions": sum(1 for d in state.decisions.values() if d.status == "invalid"),
        "tokens": {"prompt": usage.prompt, "cached": usage.cached, "completion": usage.completion,
                   "reasoning": usage.reasoning},
        "endpoint": endpoint_host(state.run.llm.get("base_url")), "resolved_models": state.resolved_models,
        "resumes": state.resumes, "first_event": events[0].ts, "last_event": events[-1].ts,
        "needs_attention": json.loads(attention.read_text()) if attention.exists() and state.finished_status
        == "needs_attention" else None,
    }


def cmd_status(args: argparse.Namespace) -> int:
    for directory in args.directories:
        print(json.dumps(status_line(Path(directory)), sort_keys=True))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    for directory in args.directories:
        summary = export(Path(directory))
        print(json.dumps({"run": Path(directory).name, "status": summary["status"],
                          "candidates": summary["candidates"]["allocated"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recital.control.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int)
        p.add_argument("--total-updates", dest="total_updates", type=int)
        p.add_argument("--max-concurrent-jobs", dest="max_concurrent_jobs", type=int)
        p.add_argument("--scheduler", choices=("local", "slurm"), default="slurm")
        p.add_argument("--no-audit", action="store_true", help="skip the terminal blind audit")

    pre = sub.add_parser("preflight")
    common(pre)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--arm", choices=ARMS)
    pre.set_defaults(func=cmd_preflight)
    run = sub.add_parser("run")
    common(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS)
    run.set_defaults(func=cmd_run)
    camp = sub.add_parser("campaign")
    common(camp)
    camp.add_argument("--output-root", type=Path, required=True)
    camp.add_argument("--arms", nargs="+", choices=ARMS, default=["no_self_validation"])
    camp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    camp.set_defaults(func=cmd_campaign)
    drill = sub.add_parser("drill-coder", help="scripted Strategist, real Coder and real training")
    common(drill)
    drill.add_argument("--output", type=Path, required=True)
    drill.add_argument("--arm", choices=ARMS)
    drill.add_argument("--drill-oracles", nargs="+", default=["malinois", "grammar_prior"],
                       help="enabled oracles of the task: the first gets the requested utility")
    drill.set_defaults(func=cmd_drill_coder)
    fixed = sub.add_parser("fixed-recipe", help="no-search control: the declared initial recipe trained once per --steps")
    common(fixed)
    fixed.add_argument("--output-root", type=Path, required=True)
    fixed.add_argument("--steps", nargs="+", type=int, required=True)
    fixed.add_argument("--seeds", nargs="+", type=int, required=True)
    fixed.set_defaults(func=cmd_fixed_recipe)
    stat = sub.add_parser("status")
    stat.add_argument("directories", nargs="+")
    stat.set_defaults(func=cmd_status)
    exp = sub.add_parser("export", help="write each cell's candidates, self-validation and decisions tables")
    exp.add_argument("directories", nargs="+")
    exp.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
