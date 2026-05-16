"""面向 DQN 预集成阶段的 RL 运行时辅助模块。"""

from .feature_config import (
    OBSERVATION_MODE_V10_BASIC,
    OBSERVATION_MODE_V11_FULL_10D,
    OBSERVATION_MODE_V11_LITE_6D,
    OBSERVATION_MODE_V11_NO_MAX_9D,
    OBSERVATION_MODE_V11_NO_PRIORITY_9D,
    OBSERVATION_MODE_V11_NO_RISK_9D,
    OBSERVATION_MODE_V11_NO_RISK_NO_UTIL_8D,
    OBSERVATION_MODE_V11_NO_UTIL_9D,
    OBSERVATION_MODE_V12_FULL_14D,
    FeatureConfig,
    supports_task_structured_features,
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
    "OBSERVATION_MODE_V11_NO_RISK_9D",
    "OBSERVATION_MODE_V11_NO_UTIL_9D",
    "OBSERVATION_MODE_V11_NO_MAX_9D",
    "OBSERVATION_MODE_V11_NO_PRIORITY_9D",
    "OBSERVATION_MODE_V11_NO_RISK_NO_UTIL_8D",
    "OBSERVATION_MODE_V11_LITE_6D",
    "OBSERVATION_MODE_V12_FULL_14D",
    "supports_task_structured_features",
]
