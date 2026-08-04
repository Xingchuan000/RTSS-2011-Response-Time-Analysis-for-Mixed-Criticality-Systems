"""Ordinary deployment replay adapter used by nonvacuity_lab HOUT pairs.

This CLI deliberately has no experiment/mutation arguments.  A runtime config
may name an ordinary ``module:function`` factory; the returned runner must
provide ``run(scenarios=...) -> (summary, events)``.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("ordinary_hout_factory must be module:function")
    return getattr(importlib.import_module(module_name), function_name)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "events.jsonl"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite HOUT output: {args.output_dir / name}")
    scenarios = json.loads(args.scenario_file.read_text(encoding="utf-8"))
    runtime_config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    factory = _load_factory(str(runtime_config["ordinary_hout_factory"]))
    runner = factory(seed_dir=args.seed_dir, tree_path=args.tree, runtime_config=runtime_config)
    summary, events = runner.run(scenarios=scenarios)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
