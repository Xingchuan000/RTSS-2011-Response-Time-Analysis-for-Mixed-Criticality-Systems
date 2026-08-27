from __future__ import annotations

from .config import ProofRoute, ProofRouteConfig
from .protected_prefix import ROUTE as PROTECTED_PREFIX_ROUTE
from .raw_protected_prefix import ROUTE as RAW_PROTECTED_PREFIX_ROUTE
from .strict_full import ROUTE as STRICT_FULL_ROUTE


def resolve_route(route: ProofRoute | ProofRouteConfig | str):
    value = route.route if isinstance(route, ProofRouteConfig) else route
    if value == ProofRoute.PROTECTED_PREFIX or value == "protected_prefix":
        return PROTECTED_PREFIX_ROUTE
    if value == ProofRoute.RAW_PROTECTED_PREFIX or value == "raw_protected_prefix":
        return RAW_PROTECTED_PREFIX_ROUTE
    if value == ProofRoute.STRICT_FULL or value == "strict_full":
        return STRICT_FULL_ROUTE
    raise ValueError("PROOF_ROUTE_INVALID")
