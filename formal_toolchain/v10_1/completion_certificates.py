"""Priority-ordered certified completion envelopes through V10.14.

The active implementation remains under ``formal_toolchain.v10_1`` because the
request/bundle schema is unchanged.  Completed BASE, pointwise-PCSSC, V10.12
case-consistent, V10.13 conditioned-carry, and V10.14 refined-case certificates
from *strictly higher-priority* tasks may be reused by a lower-priority PCSSC
target.

A certificate is deliberately stronger than a boolean target-safety result: it
carries an explicit response/completion upper bound and is accepted only when
``0 < R <= D <= T``.  The verifier constructs the map in canonical fixed-
priority order, which makes cross-target reuse acyclic by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .kernel.symbolic_state import BoundModel, TaskBound
else:
    BoundModel = Any
    TaskBound = Any


BASE_COMPLETION_SOURCE = "BASE_C_AMC_SEM_SECTION4_1_SUCCESSFUL_PREFIX"
PCSSC_COMPLETION_SOURCE = "TARGET_PROVED_BY_PCSSC"
PCSSC_POINTWISE_COMPLETION_THEOREM = "PCSSC_SAFE_PREFIX_COMPLETION_EXPORT_V10_11"
PCSSC_CASE_COMPLETION_THEOREM = "PCSSC_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_12"
PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM = (
    "PCSSC_CASE_CONDITIONED_SAFE_PREFIX_COMPLETION_EXPORT_V10_13"
)
PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14 = (
    "PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_14"
)
BASE_COMPLETION_THEOREM = "BASE_SECTION4_1_COMPLETION_EXPORT_V10_11"


class CompletionCertificateError(ValueError):
    """Fail-closed error for an invalid or cyclic completion certificate."""


@dataclass(frozen=True, slots=True)
class CertifiedCompletionBound:
    task: str
    response_bound: int
    source: str
    priority: int
    deadline: int
    period: int
    theorem_basis: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "task": self.task,
            "response_bound": int(self.response_bound),
            "source": self.source,
            "priority": int(self.priority),
            "deadline": int(self.deadline),
            "period": int(self.period),
            "theorem_basis": self.theorem_basis,
        }


def _validated_certificate(
    task: TaskBound,
    response_bound: int,
    *,
    source: str,
    theorem_basis: str,
) -> CertifiedCompletionBound:
    bound = int(response_bound)
    if bound <= 0:
        raise CompletionCertificateError(
            f"CERTIFIED_COMPLETION_BOUND_NONPOSITIVE:{task.name}:{bound}"
        )
    if int(task.deadline) > int(task.period):
        raise CompletionCertificateError(
            f"CERTIFIED_COMPLETION_REQUIRES_D_LE_T:{task.name}:"
            f"D={task.deadline}:T={task.period}"
        )
    if bound > int(task.deadline):
        raise CompletionCertificateError(
            f"CERTIFIED_COMPLETION_BOUND_EXCEEDS_DEADLINE:{task.name}:"
            f"R={bound}:D={task.deadline}"
        )
    return CertifiedCompletionBound(
        task=task.name,
        response_bound=bound,
        source=str(source),
        priority=int(task.priority),
        deadline=int(task.deadline),
        period=int(task.period),
        theorem_basis=str(theorem_basis),
    )


def build_base_completion_certificates(
    model: BoundModel,
    completion_bound_by_task: Mapping[str, int],
) -> dict[str, CertifiedCompletionBound]:
    """Validate and export the successful Section-4.1 completion prefix."""

    known = model.task_by_name
    extra = sorted(set(str(name) for name in completion_bound_by_task) - set(known))
    if extra:
        raise CompletionCertificateError(
            "CERTIFIED_COMPLETION_UNKNOWN_BASE_TASK:" + ",".join(extra)
        )
    result: dict[str, CertifiedCompletionBound] = {}
    for task in model.tasks:
        if task.name not in completion_bound_by_task:
            continue
        result[task.name] = _validated_certificate(
            task,
            int(completion_bound_by_task[task.name]),
            source=BASE_COMPLETION_SOURCE,
            theorem_basis=BASE_COMPLETION_THEOREM,
        )
    return result


def export_pcssc_completion_certificate(
    model: BoundModel,
    target_name: str,
    *,
    status: str,
    response_bound: int | None,
    theorem_basis: str,
) -> CertifiedCompletionBound:
    """Turn a completed PCSSC PASS into a downstream completion certificate."""

    if status != "PASS":
        raise CompletionCertificateError(
            f"PCSSC_COMPLETION_EXPORT_REQUIRES_PASS:{target_name}:{status}"
        )
    if response_bound is None:
        raise CompletionCertificateError(
            f"PCSSC_COMPLETION_EXPORT_MISSING_RESPONSE_BOUND:{target_name}"
        )
    task = model.task_by_name.get(target_name)
    if task is None:
        raise CompletionCertificateError(
            f"PCSSC_COMPLETION_EXPORT_UNKNOWN_TASK:{target_name}"
        )
    if task.criticality != "HI":
        raise CompletionCertificateError(
            f"PCSSC_COMPLETION_EXPORT_REQUIRES_HI:{target_name}"
        )
    if theorem_basis not in {
        PCSSC_POINTWISE_COMPLETION_THEOREM,
        PCSSC_CASE_COMPLETION_THEOREM,
        PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM,
        PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14,
    }:
        raise CompletionCertificateError(
            f"PCSSC_COMPLETION_EXPORT_UNKNOWN_THEOREM:{target_name}:{theorem_basis}"
        )
    return _validated_certificate(
        task,
        int(response_bound),
        source=PCSSC_COMPLETION_SOURCE,
        theorem_basis=theorem_basis,
    )


def merge_certified_completion(
    current: CertifiedCompletionBound | None,
    candidate: CertifiedCompletionBound,
) -> CertifiedCompletionBound:
    """Keep the tighter of two independently sound bounds for one task."""

    if current is None:
        return candidate
    if current.task != candidate.task:
        raise CompletionCertificateError(
            f"CERTIFIED_COMPLETION_MERGE_TASK_MISMATCH:{current.task}:{candidate.task}"
        )
    for field_name in ("priority", "deadline", "period"):
        if getattr(current, field_name) != getattr(candidate, field_name):
            raise CompletionCertificateError(
                f"CERTIFIED_COMPLETION_MERGE_BINDING_MISMATCH:{candidate.task}:{field_name}"
            )
    return candidate if candidate.response_bound < current.response_bound else current


def completion_prefix_for_target(
    model: BoundModel,
    target_name: str,
    certificates: Mapping[str, CertifiedCompletionBound],
) -> dict[str, CertifiedCompletionBound]:
    """Return the already-certified strict higher-priority prefix.

    The caller is expected to sweep targets in canonical priority order.  Any
    self/lower-priority entry is treated as a proof-DAG violation instead of
    being silently ignored, so a future refactor cannot accidentally introduce
    backward safety feedback.
    """

    target = model.task_by_name.get(target_name)
    if target is None:
        raise CompletionCertificateError(
            f"CERTIFIED_COMPLETION_TARGET_UNKNOWN:{target_name}"
        )
    result: dict[str, CertifiedCompletionBound] = {}
    for name, certificate in certificates.items():
        task = model.task_by_name.get(name)
        if task is None or certificate.task != name:
            raise CompletionCertificateError(
                f"CERTIFIED_COMPLETION_BINDING_MISMATCH:{name}"
            )
        # Revalidate identity fields on every use; certificates are small and
        # this keeps the proof boundary explicit.
        if (
            certificate.priority != int(task.priority)
            or certificate.deadline != int(task.deadline)
            or certificate.period != int(task.period)
            or certificate.response_bound <= 0
            or certificate.response_bound > int(task.deadline)
            or int(task.deadline) > int(task.period)
        ):
            raise CompletionCertificateError(
                f"CERTIFIED_COMPLETION_CERTIFICATE_INVALID:{name}"
            )
        if int(task.priority) >= int(target.priority):
            raise CompletionCertificateError(
                f"PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION:{name}->{target_name}"
            )
        result[name] = certificate
    return result


__all__ = [
    "BASE_COMPLETION_SOURCE",
    "PCSSC_COMPLETION_SOURCE",
    "PCSSC_POINTWISE_COMPLETION_THEOREM",
    "PCSSC_CASE_COMPLETION_THEOREM",
    "PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM",
    "CertifiedCompletionBound",
    "CompletionCertificateError",
    "build_base_completion_certificates",
    "completion_prefix_for_target",
    "export_pcssc_completion_certificate",
    "merge_certified_completion",
]
