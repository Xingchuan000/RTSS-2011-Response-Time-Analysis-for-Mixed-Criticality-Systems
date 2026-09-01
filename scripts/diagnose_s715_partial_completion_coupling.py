#!/usr/bin/env python3
"""Diagnostic-only partial completion coupling for s715 / mc_sd_hi_2.

The current V10.12 implementation uses completion-aware per-task workload only
when every higher-priority LO task has a single-job completion envelope.  This
diagnostic keeps that exact arithmetic for the certified subset and places only
the uncertified LO subset (currently mc_sd_lo_3) behind an independent safe
aggregate carry + future bound.  It does not add a proof rule or change PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict): raise ValueError(f"expected object: {path}")
    return obj


def _find_workspace(root: Path, seed: str) -> Path:
    direct = root / seed
    if (direct / "request/proof_request.json").is_file(): return direct
    rows = sorted({p.parent.parent.resolve() for p in root.rglob("request/proof_request.json") if seed in str(p.parent.parent)})
    if len(rows) != 1: raise ValueError(f"expected one workspace matching {seed!r}, found {len(rows)}")
    return rows[0]


class WorkspaceInput:
    def __init__(self, path: Path, seed: str) -> None:
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if path.is_dir(): root = path.resolve()
        elif path.is_file() and path.suffix.lower() == ".zip":
            self._tmp = tempfile.TemporaryDirectory(prefix="s715_partial_"); root = Path(self._tmp.name)
            with zipfile.ZipFile(path) as zf: zf.extractall(root)
        else: raise ValueError("--formal must be a directory or ZIP")
        self.workspace = _find_workspace(root, seed)
    def close(self):
        if self._tmp: self._tmp.cleanup()


def _rebuild_model(workspace: Path, source_root: Path):
    if str(source_root) not in sys.path: sys.path.insert(0, str(source_root))
    from formal_toolchain.v10_1.bindings import build_bindings
    from formal_toolchain.v10_1.kernel.symbolic_state import BoundModel
    return BoundModel.from_bindings(build_bindings(workspace / "request/proof_request.json", source_root=source_root), max_jobs_per_task=2)


def _rebuild_path(receipts: dict[str, Any]):
    from formal_toolchain.v10_1.controller_macro import BudgetInterval, ControllerMacroPath
    raw = receipts["controller_macro"]
    boxes = tuple({str(n): BudgetInterval(int(v["lower"]), int(v["upper"])) for n,v in b.items()} for b in raw["boxes"])
    return ControllerMacroPath(boxes=boxes, receipts=tuple(raw.get("receipts", ())), conservatism_ledger=tuple(raw.get("conservatism_ledger", ())))


def _pcssc_target(receipts: dict[str, Any], target: str) -> dict[str, Any]:
    rows=[r for r in receipts.get("pcssc_targets",[]) if isinstance(r,dict) and r.get("target")==target]
    if len(rows)!=1: raise ValueError(f"expected one PCSSC row for {target}")
    return rows[0]


def _completion_map(row: dict[str, Any], target: str) -> dict[str,int]:
    oid=f"REACHABLE_CARRY_IN::{target}"
    hits=[r for r in row.get("receipts",[]) if isinstance(r,dict) and r.get("obligation_id")==oid]
    if len(hits)!=1: raise ValueError(f"missing {oid}")
    return {str(k):int(v) for k,v in hits[0]["effective_completion_envelopes"].items()}


def _failing_case(row: dict[str, Any]) -> dict[str, Any]:
    rows=[r for r in row.get("tested_horizons",[]) if isinstance(r,dict) and r.get("terminal")=="CASE_CONSISTENT" and r.get("status")=="UNRESOLVED"]
    if not rows: raise ValueError("no unresolved case-consistent case")
    return rows[0]


def _partial_bound(model, target, path, completion: dict[str,int], *, horizon:int, theta:int, switch, classification:str):
    import formal_toolchain.v10_1.pcssc as p
    idx=next(i for i,t in enumerate(model.tasks) if t.name==target.name)
    hp=tuple(model.tasks[:idx]); protected=set(completion)
    certified=tuple(t for t in hp if not (t.criticality=="LO" and t.name not in protected))
    residual=tuple(t for t in hp if t.criticality=="LO" and t.name not in protected)
    controller_times=p.candidate_controller_times(theta, model.agent_period, horizon)
    cells=p._macro_cells(horizon, controller_times, switch)
    target_work=p._target_cap(target, classification)
    certified_total=0; cert_rows=[]
    for task in certified:
        weights=tuple(p._weight_for_cell(task,c,switch,controller_times,path) for c in cells)
        value, details=p._exact_periodic_task_workload(target,task,theta=theta,horizon=horizon,cells=cells,weights=weights,switch_kind=switch.kind,protected=protected,protected_response_by_task=completion,controller_period=model.agent_period)
        certified_total += int(value); cert_rows.append({"task":task.name,"bound":int(value),**details})
    residual_carry=0; carry_details={}
    if residual:
        specs=p._carry_task_specs(residual,path)
        if switch.kind=="PRE_HI":
            residual_carry, carry_details=p.phase_relaxed_single_switch_carry(int(target.period),int(model.agent_period),int(theta),specs)
        else:
            residual_carry, carry_details=p.phase_relaxed_lo_entry_carry(int(target.period),int(model.agent_period),int(theta),specs)
    residual_future=0; residual_rows=[]
    for task in residual:
        weights=tuple(p._weight_for_cell(task,c,switch,controller_times,path) for c in cells)
        value, details=p._exact_periodic_task_future_only(target,task,theta=theta,horizon=horizon,cells=cells,weights=weights,controller_period=model.agent_period)
        residual_future += int(value); residual_rows.append({"task":task.name,"future_bound":int(value),**details})
    partial_hp=int(certified_total)+int(residual_carry)+int(residual_future)
    current_w,current_details=p._workload_case(model,target,hp,path,protected,completion,horizon=horizon,theta=theta,switch=switch,classification=classification)
    return int(target_work+partial_hp), {
        "target_demand":int(target_work),"certified_subset":[t.name for t in certified],"residual_uncertified_lo":[t.name for t in residual],
        "certified_subset_total":int(certified_total),"residual_carry":int(residual_carry),"residual_future":int(residual_future),"partial_hp_bound":int(partial_hp),
        "current_v10_12_W":int(current_w),"current_v10_12_selected_hp_bound":current_details.get("selected_hp_bound"),
        "current_aggregate_hp_bound":current_details.get("aggregate_hp_bound"),"current_completion_phase_coupled_hp_bound":current_details.get("completion_phase_coupled_hp_bound"),
        "certified_rows":cert_rows,"residual_rows":residual_rows,"residual_carry_details":carry_details,
    }


def _iterate(model,target,path,completion,theta,switch,classification):
    R=max(1,int(target.actual_demand_upper)); D=int(target.deadline); rows=[]; seen=set()
    while True:
        if R in seen: return None,rows,"CYCLE"
        seen.add(R)
        W,details=_partial_bound(model,target,path,completion,horizon=R,theta=theta,switch=switch,classification=classification)
        rows.append({"R":R,"W_partial":W,"postfixed":W<=R,**{k:details[k] for k in ["certified_subset_total","residual_carry","residual_future","partial_hp_bound","current_v10_12_W"]}})
        if W<=R: return R,rows,None
        if R==D: return None,rows,f"NO_POSTFIX:W={W}:D={D}"
        R=min(D,max(R+1,W))


def _markdown(report):
    lines=["# s715 partial completion-coupling diagnostic","",f"Signal: **{report['diagnostic_signal']}**","",f"Uncertified LO residual: `{', '.join(report['residual_uncertified_lo'])}`","", "| R | current W | partial W | certified subset | residual carry | residual future |", "|---:|---:|---:|---:|---:|---:|"]
    for r in report["partial_recurrence"]:
        lines.append(f"| {r['R']} | {r['current_v10_12_W']} | {r['W_partial']} | {r['certified_subset_total']} | {r['residual_carry']} | {r['residual_future']} |")
    lines += ["", "This is a diagnostic bound only; it is not wired into the formal PASS route."]
    return "\n".join(lines)+"\n"


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--formal",required=True,type=Path); ap.add_argument("--source-root",required=True,type=Path)
    ap.add_argument("--seed-workspace",default="s715_best_overall_v9_1_e2e"); ap.add_argument("--target",default="mc_sd_hi_2"); ap.add_argument("--output-dir",type=Path,default=Path("diagnostic_s715_partial_completion_coupling")); args=ap.parse_args()
    wi=WorkspaceInput(args.formal,args.seed_workspace)
    try:
        receipts=_read_json(wi.workspace/"verified/proof_receipts.json"); row=_pcssc_target(receipts,args.target); completion=_completion_map(row,args.target)
        model=_rebuild_model(wi.workspace,args.source_root.resolve()); path=_rebuild_path(receipts); target=model.task_by_name[args.target]
        case=_failing_case(row)
        import formal_toolchain.v10_1.pcssc as p
        switch=p.SwitchCell(str(case["switch_kind"]),case.get("switch_lower"),case.get("switch_upper")); theta=int(case["theta"]); classification=str(case["target_classification"])
        rcert,path_rows,failure=_iterate(model,target,path,completion,theta,switch,classification)
        idx=next(i for i,t in enumerate(model.tasks) if t.name==target.name); hp=tuple(model.tasks[:idx]); residual=[t.name for t in hp if t.criticality=="LO" and t.name not in completion]
        if rcert is not None and rcert<=int(target.deadline): signal="PARTIAL_COUPLING_SUFFICIENT_TO_CLOSE_DIAGNOSTIC"
        else: signal="PARTIAL_COUPLING_STILL_INSUFFICIENT"
        report={"schema_version":"s715_partial_completion_coupling_v1","diagnostic_only":True,"target":args.target,"deadline":int(target.deadline),"case_id":case.get("case_id"),"effective_completion_envelopes":completion,"residual_uncertified_lo":residual,"partial_response_candidate":rcert,"failure":failure,"diagnostic_signal":signal,"partial_recurrence":path_rows,"formal_warning":"not a V10.12 proof rule; use only to decide whether a partial-coupling theorem is worth formalizing"}
        args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"s715_partial_completion_coupling.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); (args.output_dir/"s715_partial_completion_coupling.md").write_text(_markdown(report),encoding="utf-8")
        print(json.dumps({"signal":signal,"partial_response_candidate":rcert,"residual_lo":residual},ensure_ascii=False)); return 0
    finally: wi.close()

if __name__=="__main__": raise SystemExit(main())
