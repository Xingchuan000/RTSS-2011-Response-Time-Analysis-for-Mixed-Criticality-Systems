"""Complete recurring protected input oracle.

The oracle is a deterministic projection of a full-reference input oracle.
It validates every queried recurring job before caching it so malformed or
mode-dependent inputs cannot silently become proof witnesses.

Phase C additions:
    - AuthoritativeFullExecutionInput Protocol (Section 5.2)
    - FullJobInput frozen dataclass
    - project_full_input_oracle() lazy infinite projection (Section 5.3)
    - check_demand_receptiveness() theorem (Section 5.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from formal_toolchain.core.hashing import sha256_object

from .types import ProtectedPrefixBuildResult

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class FullJobInput:
    """Fixed input record for one job in a full execution.

    Fields match Section 5.2 of the PP0 proof plan.
    """
    job_key: tuple[str, int]
    release_time: int
    actual_demand: int
    hi_class: Literal["NORMAL", "ABNORMAL"] | None


class AuthoritativeFullExecutionInput(Protocol):
    """The authoritative recurring release-fixed input oracle of a single
    full-reference execution (Section 5.2).

    The oracle fixes actual_demand for every job; it must NOT regenerate
    demand from WCET or re-sample a random demand function.  The same
    (task_name, release_index) must always return the same FullJobInput.
    """

    def input_for(self, task_name: str, release_index: int) -> FullJobInput:
        ...

    def oracle_fingerprint(self) -> str:
        ...


class FullReferenceInputOracle(Protocol):
    """Authoritative release-fixed input stream of one full execution."""

    def input_for(self, task_name: str, release_index: int) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ProtectedJobInput:
    job_key: JobKey
    release_time: int
    actual_demand: int
    hi_class: Literal["NORMAL", "ABNORMAL"] | None


class FullReferenceRecurringInputOracle:
    """Parameterized recurring input oracle for the full reference taskset.

    Computes each job's release time, actual demand, and HI class from task
    parameters.  The oracle is deterministic: inserting, deleting, or
    reordering queries does not change any individual job record.

    Guarantees:
    - release_time = offset_i + q * period_i
    - q >= 0
    - actual_demand > 0
    - HI normal:   demand <= C_i(LO)
    - HI abnormal: C_i(LO) < demand <= C_i(HI)
    - LO:          demand <= C_i(LO)  (full-reference bound)
    - Same (task_name, q) always returns the same record.
    """

    def __init__(
        self,
        taskset: object,
        *,
        normal_hi_demand: int = 0,
        abnormal_hi_demand: int = 0,
        lo_demand: int = 0,
        abnormal_hi_releases: frozenset[tuple[str, int]] = frozenset(),
    ) -> None:
        self._tasks = {str(task.name): task for task in taskset.tasks}
        self._normal_hi_demand = normal_hi_demand
        self._abnormal_hi_demand = abnormal_hi_demand
        self._lo_demand = lo_demand
        self._abnormal_hi_releases = abnormal_hi_releases
        self._cache: dict[JobKey, dict[str, Any]] = {}

    def input_for(self, task_name: str, release_index: int) -> dict[str, Any]:
        key: JobKey = (task_name, release_index)
        if key in self._cache:
            return self._cache[key]

        task = self._tasks.get(task_name)
        if task is None:
            raise ValueError(f"FULL_ORACLE_TASK_NOT_FOUND:{task_name}")
        if isinstance(release_index, bool) or not isinstance(release_index, int) or release_index < 0:
            raise ValueError("FULL_ORACLE_RELEASE_INDEX_INVALID")

        release_time = int(task.offset) + release_index * int(task.period)
        is_abnormal = key in self._abnormal_hi_releases

        if task.criticality == "LO":
            hi_class = None
            demand = self._lo_demand if self._lo_demand > 0 else int(task.c_lo)
            bound = int(task.c_lo)
        elif is_abnormal:
            hi_class = "ABNORMAL"
            demand = self._abnormal_hi_demand if self._abnormal_hi_demand > 0 else int(task.c_hi)
            bound = int(task.c_hi)
            if demand <= int(task.c_lo):
                raise ValueError(f"ABNORMAL_HI_DEMAND_NOT_ABOVE_C_LO:{task_name}:{release_index}")
        else:
            hi_class = "NORMAL"
            demand = self._normal_hi_demand if self._normal_hi_demand > 0 else int(task.c_lo)
            bound = int(task.c_lo)

        if demand <= 0 or demand > bound:
            raise ValueError(f"FULL_ORACLE_DEMAND_OUT_OF_BOUND:{task_name}:{release_index}:{demand}:{bound}")

        result = {
            "job_key": key,
            "task_name": task_name,
            "release_index": release_index,
            "release_time": release_time,
            "actual_demand": demand,
            "hi_class": hi_class,
        }
        self._cache[key] = result
        return result

    def oracle_fingerprint(self) -> str:
        """Identify the fixed stream definition, independently of query order."""
        return sha256_object({
            "kind": type(self).__name__,
            "tasks": sorted((name, int(task.offset), int(task.period),
                              int(task.c_lo), int(task.c_hi), str(task.criticality))
                             for name, task in self._tasks.items()),
            "normal_hi_demand": self._normal_hi_demand,
            "abnormal_hi_demand": self._abnormal_hi_demand,
            "lo_demand": self._lo_demand,
            "abnormal_hi_releases": sorted(self._abnormal_hi_releases),
        })


class ProtectedInputOracle:
    """Validated projection of full-reference inputs to the protected prefix.

    This object is an input contract, not a proof of complete-execution
    existence.  It guarantees only that each queried protected job has a
    stable, release-fixed and WCET-legal input record.
    """

    def __init__(
        self,
        full_oracle: FullReferenceInputOracle,
        protected_task_names: frozenset[str],
        construction: ProtectedPrefixBuildResult,
    ) -> None:
        expected = frozenset(construction.protected_task_names)
        if protected_task_names != expected:
            raise ValueError("PROTECTED_INPUT_ORACLE_PARTITION_MISMATCH")
        self._full = full_oracle
        self._protected = protected_task_names
        self._construction = construction
        self._tasks = {str(task.name): task for task in construction.prefix_taskset.tasks}
        self._cache: dict[JobKey, ProtectedJobInput] = {}

    def oracle_fingerprint(self) -> str:
        parent = getattr(self._full, "oracle_fingerprint", None)
        return sha256_object({
            "kind": type(self).__name__,
            "parent": parent() if callable(parent) else None,
            "protected_task_names": sorted(self._protected),
            "prefix_fingerprint": self._construction.prefix_taskset.to_dict()["fingerprint"],
        })

    def input_for(self, task_name: str, release_index: int) -> ProtectedJobInput:
        if task_name not in self._protected or task_name not in self._tasks:
            raise ValueError(f"PROTECTED_INPUT_TASK_OUTSIDE_PREFIX:{task_name}")
        if isinstance(release_index, bool) or not isinstance(release_index, int) or release_index < 0:
            raise ValueError("PROTECTED_INPUT_RELEASE_INDEX_INVALID")

        key: JobKey = (task_name, release_index)
        if key in self._cache:
            return self._cache[key]

        full_input = self._full.input_for(task_name, release_index)
        if not isinstance(full_input, Mapping):
            raise ValueError("FULL_INPUT_ORACLE_RECORD_INVALID")
        declared_key = full_input.get("job_key")
        if declared_key is not None and tuple(declared_key) != key:
            raise ValueError("FULL_INPUT_ORACLE_JOB_KEY_MISMATCH")

        release_time = full_input.get("release_time")
        demand = full_input.get("actual_demand")
        hi_class = full_input.get("hi_class")
        if isinstance(release_time, bool) or not isinstance(release_time, int) or release_time < 0:
            raise ValueError("FULL_INPUT_ORACLE_RELEASE_TIME_INVALID")
        if isinstance(demand, bool) or not isinstance(demand, int) or demand <= 0:
            raise ValueError("FULL_INPUT_ORACLE_DEMAND_INVALID")

        task = self._tasks[task_name]
        expected_release_time = int(task.offset) + release_index * int(task.period)
        if release_time != expected_release_time:
            raise ValueError("FULL_INPUT_ORACLE_RELEASE_TIME_NOT_PERIODIC")
        if task.criticality == "LO":
            if hi_class is not None:
                raise ValueError("LO_INPUT_MUST_NOT_HAVE_HI_CLASS")
            bound = int(task.c_lo)  # saturated prefix has C_LO == C_HI for LO tasks
        else:
            if hi_class not in {"NORMAL", "ABNORMAL"}:
                raise ValueError("HI_INPUT_CLASS_INVALID")
            if hi_class == "NORMAL":
                bound = int(task.c_lo)
            else:
                if demand <= int(task.c_lo):
                    raise ValueError("ABNORMAL_HI_DEMAND_NOT_ABOVE_C_LO")
                bound = int(task.c_hi)
        if demand > bound:
            raise ValueError("PROTECTED_INPUT_DEMAND_EXCEEDS_REFERENCE_BOUND")

        result = ProtectedJobInput(
            job_key=key,
            release_time=release_time,
            actual_demand=demand,
            hi_class=hi_class,
        )
        self._cache[key] = result
        return result


def project_full_oracle_to_prefix(
    full_oracle: FullReferenceInputOracle,
    protected_task_names: frozenset[str],
    construction: ProtectedPrefixBuildResult,
) -> ProtectedInputOracle:
    """Project a full-reference input oracle to a protected-prefix oracle."""
    return ProtectedInputOracle(full_oracle, protected_task_names, construction)


# ---------------------------------------------------------------------------
# Phase C: Authoritative recurring input projection (Section 5.3)
# ---------------------------------------------------------------------------


class LazyInfiniteProtectedInputOracle:
    """Lazy infinite projection of an authoritative full-execution input oracle.

    Section 5.3 requirements:
    - job key is preserved
    - release_time is preserved
    - actual_demand is preserved
    - HI normal/abnormal class is preserved
    - tail job queries are rejected
    - repeated queries return the same result
    - query order does not affect results
    """

    def __init__(
        self,
        full_input_oracle: AuthoritativeFullExecutionInput,
        construction: ProtectedPrefixBuildResult,
    ) -> None:
        self._full = full_input_oracle
        self._protected = frozenset(construction.protected_task_names)
        self._construction = construction
        self._cache: dict[JobKey, ProtectedJobInput] = {}
        self._tasks = {
            str(task.name): task for task in construction.prefix_taskset.tasks
        }
        self._demand_check_cache: list[dict[str, Any]] = []

    def oracle_fingerprint(self) -> str:
        parent = getattr(self._full, "oracle_fingerprint", None)
        parent_fp = parent() if callable(parent) else None
        return sha256_object({
            "kind": type(self).__name__,
            "full_oracle_fingerprint": parent_fp,
            "protected_task_names": sorted(self._protected),
            "construction_fingerprint": self._construction.prefix_taskset.to_dict()["fingerprint"],
        })

    @property
    def execution_id(self) -> str:
        return str(getattr(self._full, "execution_id", "arbitrary-full-execution"))

    def canonical_protected_batch_order(
        self, task_names: tuple[str, ...] | list[str], release_index: int,
    ) -> tuple[JobKey, ...]:
        """Canonical protected order used by executable ARR batch semantics."""
        keys = [(str(name), int(release_index)) for name in task_names
                if str(name) in self._protected]
        return tuple(sorted(keys, key=lambda key: (
            int(self._tasks[key[0]].priority_index), key[0], key[1],
        )))

    def input_for(self, task_name: str, release_index: int) -> ProtectedJobInput:
        if task_name not in self._protected:
            raise ValueError(f"PROTECTED_INPUT_TASK_OUTSIDE_PREFIX:{task_name}")
        if task_name not in self._tasks:
            raise ValueError(f"PROTECTED_INPUT_TASK_OUTSIDE_PREFIX_TASKSET:{task_name}")
        if not isinstance(release_index, int) or isinstance(release_index, bool) or release_index < 0:
            raise ValueError("PROTECTED_INPUT_RELEASE_INDEX_INVALID")

        key: JobKey = (task_name, release_index)
        if key in self._cache:
            return self._cache[key]

        provider = getattr(self._full, "input_for", None) or getattr(self._full, "record_for", None)
        if not callable(provider):
            raise ValueError("FULL_EXECUTION_RELEASE_LEDGER_PROVIDER_MISSING")
        full_input = provider(task_name, release_index)
        if isinstance(full_input, Mapping):
            try:
                full_input = FullJobInput(
                    job_key=tuple(full_input["job_key"]),
                    release_time=int(full_input["release_time"]),
                    actual_demand=int(full_input["actual_demand"]),
                    hi_class=full_input.get("hi_class"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("FULL_INPUT_NOT_FULL_JOB_INPUT") from exc
        if not isinstance(full_input, FullJobInput):
            raise ValueError("FULL_INPUT_NOT_FULL_JOB_INPUT")

        declared_key = full_input.job_key
        if declared_key != key:
            raise ValueError("FULL_INPUT_JOB_KEY_MISMATCH")

        release_time = full_input.release_time
        demand = full_input.actual_demand
        hi_class = full_input.hi_class

        if not isinstance(release_time, int) or release_time < 0:
            raise ValueError("FULL_INPUT_RELEASE_TIME_INVALID")
        if not isinstance(demand, int) or demand <= 0:
            raise ValueError("FULL_INPUT_DEMAND_INVALID")

        task = self._tasks[task_name]
        expected_release_time = int(task.offset) + release_index * int(task.period)
        if release_time != expected_release_time:
            raise ValueError("FULL_INPUT_RELEASE_TIME_NOT_PERIODIC")

        if task.criticality == "LO":
            if hi_class is not None:
                raise ValueError("LO_INPUT_MUST_NOT_HAVE_HI_CLASS")
            bound = int(task.c_lo)
        else:
            if hi_class not in {"NORMAL", "ABNORMAL"}:
                raise ValueError("HI_INPUT_CLASS_INVALID")
            if hi_class == "NORMAL":
                bound = int(task.c_lo)
            else:
                if demand <= int(task.c_lo):
                    raise ValueError("ABNORMAL_HI_DEMAND_NOT_ABOVE_C_LO")
                bound = int(task.c_hi)
        if demand > bound:
            raise ValueError("PROTECTED_INPUT_DEMAND_EXCEEDS_REFERENCE_BOUND")

        result = ProtectedJobInput(
            job_key=key,
            release_time=release_time,
            actual_demand=demand,
            hi_class=hi_class,
        )
        self._cache[key] = result
        return result

    def project_all_protected_inputs_up_to(
        self, release_index_bound: int
    ) -> tuple[ProtectedJobInput, ...]:
        """Project all protected inputs up to a given release index bound.

        This provides a finite diagnostic view only.  It cannot prove
        the universal theorem over all recurring releases.  The theorem
        is discharged by the parameterized demand receptiveness kernel.
        """
        results: list[ProtectedJobInput] = []
        for name in sorted(self._protected):
            for q in range(release_index_bound + 1):
                results.append(self.input_for(name, q))
        return tuple(results)

    def demand_receptiveness_checks(
        self, release_index_bound: int,
        *,
        proof_kernel_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute demand receptiveness for all protected jobs up to horizon.

        Section 5.4:
        - For LO: demand <= C_ref_LO = C_pp_LO = C_pp_HI
        - For HI NORMAL: demand <= C_pp_LO
        - For HI ABNORMAL: C_pp_LO < demand <= C_pp_HI
        """
        checks: list[dict[str, Any]] = []
        for name in sorted(self._protected):
            task = self._tasks[name]
            c_lo = int(task.c_lo)
            c_hi = int(task.c_hi)
            for q in range(release_index_bound + 1):
                inp = self.input_for(name, q)
                job_check = {
                    "task_name": name,
                    "release_index": q,
                    "actual_demand": inp.actual_demand,
                    "hi_class": inp.hi_class,
                }
                if task.criticality == "LO":
                    job_check["legal"] = inp.actual_demand <= c_lo
                    job_check["bound"] = c_lo
                    job_check["rule"] = "A <= C_ref_LO = C_pp_LO = C_pp_HI"
                elif inp.hi_class == "NORMAL":
                    job_check["legal"] = inp.actual_demand <= c_lo
                    job_check["bound"] = c_lo
                    job_check["rule"] = "NORMAL => A <= C_pp_LO"
                elif inp.hi_class == "ABNORMAL":
                    job_check["legal"] = c_lo < inp.actual_demand <= c_hi
                    job_check["bound"] = c_hi
                    job_check["rule"] = "ABNORMAL => C_pp_LO < A <= C_pp_HI"
                else:
                    job_check["legal"] = False
                    job_check["rule"] = "UNKNOWN_CLASS"
                checks.append(job_check)

        # These rows are finite diagnostics only.  They cannot establish the
        # theorem quantified over every recurring release index.  PASS requires
        # an independently verified parametric proof receipt bound to this exact
        # projected oracle and to the saturation construction.
        all_ok = all(c["legal"] for c in checks)
        oracle_fp = self.oracle_fingerprint()
        construction_fp = self._construction.prefix_taskset.to_dict()["fingerprint"]
        kernel_ok = (
            isinstance(proof_kernel_receipt, Mapping)
            and proof_kernel_receipt.get("status") == "PASS"
            and proof_kernel_receipt.get("theorem_id")
                == "PROTECTED_INPUT_DEMAND_RECEPTIVENESS"
            and proof_kernel_receipt.get("projected_oracle_fingerprint") == oracle_fp
            and proof_kernel_receipt.get("prefix_taskset_fingerprint") == construction_fp
            and proof_kernel_receipt.get("forall_release_indices") is True
            and proof_kernel_receipt.get("all_protected_tasks_covered") is True
            and proof_kernel_receipt.get("mode_independent_lo_saturation") is True
        )
        status = "PASS" if all_ok and kernel_ok else ("FAIL" if not all_ok else "UNRESOLVED")
        return {
            "obligation_id": "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "release_index_bound": release_index_bound,
            "job_count": len(checks),
            "all_legal": all_ok,
            "finite_check_status": "PASS" if all_ok else "FAIL",
            "parameterized": kernel_ok,
            "universal_inequalities_verified": kernel_ok,
            "projected_oracle_fingerprint": oracle_fp,
            "prefix_taskset_fingerprint": construction_fp,
            "status": status,
            "code": (None if status == "PASS" else
                     "PROJECTED_DEMAND_ILLEGAL" if status == "FAIL" else
                     "PARAMETRIC_DEMAND_RECEPTIVENESS_PROOF_MISSING"),
            "checks": checks,
            "proof_kernel_receipt": dict(proof_kernel_receipt or {}),
            "certificate_hash": sha256_object({
                "checks": checks, "all_ok": all_ok,
                "oracle_fingerprint": oracle_fp,
                "prefix_taskset_fingerprint": construction_fp,
                "kernel_ok": kernel_ok,
            }),
        }


def project_full_input_oracle(
    full_input: AuthoritativeFullExecutionInput,
    construction: ProtectedPrefixBuildResult,
) -> LazyInfiniteProtectedInputOracle:
    """Project a full input oracle to protected prefix (Section 5.3)."""
    return LazyInfiniteProtectedInputOracle(full_input, construction)


def project_full_execution_ledger(
    full_execution_ledger: AuthoritativeFullExecutionInput,
    construction: ProtectedPrefixBuildResult,
) -> LazyInfiniteProtectedInputOracle:
    """Project one selected full-execution ledger without WCET regeneration."""
    if not (callable(getattr(full_execution_ledger, "input_for", None))
            or callable(getattr(full_execution_ledger, "record_for", None))):
        raise ValueError("FULL_EXECUTION_RELEASE_LEDGER_RECORD_FOR_REQUIRED")
    if not callable(getattr(full_execution_ledger, "oracle_fingerprint", None)):
        raise ValueError("FULL_EXECUTION_RELEASE_LEDGER_FINGERPRINT_REQUIRED")
    return LazyInfiniteProtectedInputOracle(full_execution_ledger, construction)
