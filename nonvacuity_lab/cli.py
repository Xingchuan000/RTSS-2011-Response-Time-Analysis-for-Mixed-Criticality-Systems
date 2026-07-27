"""CLI for the isolated PPP non-vacuity laboratory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activation import solve_symbolic_activation
from .manifest import load_campaign, load_mutation
from .preflight import audit_campaign
from .reporting.markdown_report import write_campaign_report
from .runners.campaign import run_campaign, run_one
from .schema import ExperimentStatus, experiment_envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="完全隔离、默认关闭的 PPP 非空洞性实验框架"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行 campaign")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--enable", action="store_true")
    run_parser.add_argument("--timeout-seconds", type=int)

    preflight_parser = subparsers.add_parser(
        "preflight", help="只读检查 campaign 能力和输入，不创建工作区"
    )
    preflight_parser.add_argument("--config", required=True, type=Path)
    preflight_parser.add_argument("--out", type=Path)

    one_parser = subparsers.add_parser("run-one", help="运行单个 mutation")
    one_parser.add_argument("--manifest", required=True, type=Path)
    one_parser.add_argument("--campaign-id", default="ppp_nonvacuity_one")
    one_parser.add_argument("--output-root", type=Path, default=Path("outputs/nonvacuity"))
    one_parser.add_argument("--source-root", type=Path, default=Path.cwd())
    one_parser.add_argument("--enable", action="store_true")
    one_parser.add_argument("--timeout-seconds", type=int)

    activate_parser = subparsers.add_parser("activate", help="只做 symbolic activation")
    activate_parser.add_argument("--manifest", required=True, type=Path)
    activate_parser.add_argument("--out", type=Path)
    activate_parser.add_argument("--enable", action="store_true")

    report_parser = subparsers.add_parser("report", help="重新生成 campaign 报告")
    report_parser.add_argument("--campaign-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            config = load_campaign(args.config)
            result = audit_campaign(config)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        elif args.command == "run":
            config = load_campaign(args.config)
            result = run_campaign(
                config,
                enabled_by_cli=args.enable,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "run-one":
            manifest = load_mutation(args.manifest)
            result = run_one(
                manifest,
                campaign_id=args.campaign_id,
                output_root=args.output_root.resolve(),
                source_root=args.source_root.resolve(),
                enabled_by_cli=args.enable,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "activate":
            manifest = load_mutation(args.manifest)
            if not manifest.enabled or not args.enable:
                result = experiment_envelope(
                    mutation_id=manifest.mutation_id,
                    status=ExperimentStatus.EXPERIMENT_DISABLED.value,
                )
            else:
                activation = solve_symbolic_activation(
                    mutation_id=manifest.mutation_id,
                    rule=manifest.activation,
                    output_path=args.out,
                )
                result = experiment_envelope(
                    mutation_id=manifest.mutation_id,
                    status=activation.status.value,
                    activation=activation.to_dict(),
                )
        else:
            campaign_dir = args.campaign_dir.resolve()
            result_path = campaign_dir / "campaign_result.json"
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("campaign_result.json 必须为 object")
            write_campaign_report(campaign_dir / "report.md", raw)
            result = {
                "status": "REPORT_WRITTEN",
                "path": str(campaign_dir / "report.md"),
            }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = experiment_envelope(
            status=ExperimentStatus.SETUP_INVALID.value,
            reason=str(exc),
        )
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return _exit_code(str(result.get("status")))


def _exit_code(status: str) -> int:
    if status in {
        "COMPLETED",
        "REPORT_WRITTEN",
        ExperimentStatus.EXPERIMENT_DISABLED.value,
        ExperimentStatus.PASS_EXPECTED.value,
        ExperimentStatus.FAIL_EXPECTED.value,
        ExperimentStatus.INTEGRITY_REJECTION_EXPECTED.value,
        "PASS",
    }:
        return 0
    if status == ExperimentStatus.NOT_ACTIVATED.value:
        return 3
    return 2
