"""面向 DQN 预集成阶段的 RL 运行时辅助模块。"""

from .feature_config import (
    OBSERVATION_MODE_V10_BASIC,
    OBSERVATION_MODE_V11_FULL_10D,
    FeatureConfig,
)
from .feature_state import RuntimeFeatureState
from .types import AgentObservation, AgentStepResult

__all__ = [
    "AgentObservation",
    "AgentStepResult",
    "FeatureConfig",
    "RuntimeFeatureState",
    "OBSERVATION_MODE_V10_BASIC",
    "OBSERVATION_MODE_V11_FULL_10D",
]
