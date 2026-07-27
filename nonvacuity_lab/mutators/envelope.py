"""D1 certified-envelope single-coordinate mutation."""

from __future__ import annotations

from .action_config import JsonPatchMutation, get_pointer
from .base import MutationContext, MutationResult, PreflightResult


class EnvelopeMutation(JsonPatchMutation):
    def preflight(self, context: MutationContext) -> PreflightResult:
        result = super().preflight(context)
        if result.status != "PASS":
            return result
        delta = context.parameters.get("delta")
        if not isinstance(delta, int) or isinstance(delta, bool) or delta <= 0:
            return PreflightResult("FAIL", {"reason": "envelope delta 必须为正整数"})
        return result

    def apply(self, context: MutationContext) -> MutationResult:
        path = self._target(context)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        pointer = str(context.parameters["json_pointer"])
        before = get_pointer(data, pointer)
        if not isinstance(before, int) or isinstance(before, bool):
            raise ValueError("envelope coordinate 必须为整数")
        parameters = dict(context.parameters)
        parameters["value"] = before + int(parameters["delta"])
        parameters["expected_before"] = before
        delegated = MutationContext(
            mutation_id=context.mutation_id,
            source_root=context.source_root,
            mutated_seed=context.mutated_seed,
            source_overlay=context.source_overlay,
            parameters=parameters,
        )
        result = super().apply(delegated)
        return MutationResult(
            **{
                **result.__dict__,
                "details": {
                    **dict(result.details),
                    "delta": int(parameters["delta"]),
                    "before_upper": before,
                    "after_upper": before + int(parameters["delta"]),
                },
            }
        )
