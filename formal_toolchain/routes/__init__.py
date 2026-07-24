"""Proof-route configuration and terminal-route interfaces."""

from .config import ProofRoute, ProofRouteConfig, parse_proof_route
from .resolver import resolve_route

__all__ = ["ProofRoute", "ProofRouteConfig", "parse_proof_route", "resolve_route"]
