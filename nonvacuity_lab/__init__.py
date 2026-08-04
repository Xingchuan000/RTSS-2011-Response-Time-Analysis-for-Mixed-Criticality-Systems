"""Isolated PPP non-vacuity experiment framework.

This package may call the ordinary proof/runtime packages.  The reverse import
direction is forbidden and covered by regression tests.
"""

from .schema import (
    ActivationStatus,
    ArtifactClass,
    ConfigKind,
    ExperimentStatus,
    ExpectedResult,
    MutationClass,
)

__all__ = [
    "ActivationStatus",
    "ArtifactClass",
    "ConfigKind",
    "ExperimentStatus",
    "ExpectedResult",
    "MutationClass",
]

__version__ = "1.0.0"
