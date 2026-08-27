"""Single source of truth for proof-route request parsing.

The route is proof-workflow metadata only.  It is deliberately not part of
any target/runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ProofRoute(str, Enum):
    PROTECTED_PREFIX = "protected_prefix"
    RAW_PROTECTED_PREFIX = "raw_protected_prefix"
    STRICT_FULL = "strict_full"


@dataclass(frozen=True, slots=True)
class ProofRouteConfig:
    schema_version: str
    route: ProofRoute

    @classmethod
    def default(cls) -> "ProofRouteConfig":
        return cls("proof_route_config_v1", ProofRoute.PROTECTED_PREFIX)

    def to_dict(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "route": self.route.value}


def parse_proof_route(request: Mapping[str, Any]) -> ProofRouteConfig:
    """Parse route metadata, preserving v2's historical strict semantics."""

    schema = request.get("schema_version")
    if schema == "proof_request_v2":
        return ProofRouteConfig("proof_route_config_v1", ProofRoute.STRICT_FULL)
    if schema != "proof_request_v3":
        raise ValueError("PROOF_ROUTE_REQUEST_MISSING")
    raw = request.get("proof_route")
    if not isinstance(raw, Mapping):
        raise ValueError("PROOF_ROUTE_REQUEST_MISSING")
    if raw.get("schema_version") != "proof_route_config_v1":
        raise ValueError("PROOF_ROUTE_INVALID")
    try:
        route = ProofRoute(str(raw.get("route")))
    except (TypeError, ValueError) as exc:
        raise ValueError("PROOF_ROUTE_INVALID") from exc
    return ProofRouteConfig("proof_route_config_v1", route)


def route_config(value: str | ProofRoute | None) -> ProofRouteConfig:
    if value is None:
        return ProofRouteConfig.default()
    try:
        return ProofRouteConfig("proof_route_config_v1", ProofRoute(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("PROOF_ROUTE_INVALID") from exc
