"""奖励模式配置加载器。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import ast


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """单个奖励模式的参数集合。"""

    mode: str
    job_start: float
    lo_overrun: float
    hi_overrun: float

    def describe(self) -> str:
        """返回可写入元数据/日志的配置摘要。"""

        return (
            f"{self.mode}: "
            f"job_start={self.job_start:+.6g}, "
            f"lo_overrun={self.lo_overrun:+.6g}, "
            f"hi_overrun={self.hi_overrun:+.6g}"
        )


@dataclass(frozen=True, slots=True)
class RewardModeConfig:
    """完整奖励模式配置（参数 + 公式）。"""

    mode: str
    weights: RewardWeights
    # 事件级奖励合成公式，通常把三个事件分量加总为 paper_reward。
    paper_reward_formula: str
    # 单步最终奖励公式，可引用 paper_reward/noop_bonus/budget_change_penalty 等上下文变量。
    step_reward_formula: str
    # 奖励公式常量参数（例如 noop_bonus / budget_change_penalty / budget_drift_penalty）。
    reward_parameters: dict[str, float]

    def describe(self) -> str:
        """返回用于日志/元数据的人类可读描述。"""

        return (
            f"{self.mode}: paper={self.paper_reward_formula}; "
            f"step={self.step_reward_formula}; "
            f"{self.weights.describe()}; "
            f"params={self.reward_parameters}"
        )


def reward_config_dir() -> Path:
    """返回奖励模式配置目录。"""

    return Path(__file__).resolve().parents[2] / "configs" / "reward_modes"


@lru_cache(maxsize=1)
def available_reward_modes() -> tuple[str, ...]:
    """扫描并返回可用奖励模式名。"""

    config_dir = reward_config_dir()
    return tuple(sorted(path.stem for path in config_dir.glob("*.json") if path.is_file()))


@lru_cache(maxsize=None)
def load_reward_weights(mode: str) -> RewardWeights:
    """兼容旧调用：按模式名读取奖励权重。"""

    return load_reward_mode_config(mode).weights


@lru_cache(maxsize=None)
def load_reward_mode_config(mode: str) -> RewardModeConfig:
    """按模式名读取奖励模式配置文件。"""

    config_path = reward_config_dir() / f"{mode}.json"
    if not config_path.exists():
        available = ", ".join(available_reward_modes()) or "<none>"
        raise ValueError(f"Unknown reward mode: {mode}. Available modes: {available}")

    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # 兼容两种配置写法：
    # 1) 新写法：{"event_weights": {...}, "paper_reward_formula": "...", "step_reward_formula": "..."}
    # 2) 旧写法：{"job_start":..., "lo_overrun":..., "hi_overrun":...}
    raw_weights = payload.get("event_weights", payload)
    required_keys = ("job_start", "lo_overrun", "hi_overrun")
    missing_keys = [key for key in required_keys if key not in raw_weights]
    if missing_keys:
        raise ValueError(
            f"Reward config {config_path} missing weight keys: {missing_keys}. "
            f"Required keys are: {required_keys}"
        )

    weights = RewardWeights(
        mode=mode,
        job_start=float(raw_weights["job_start"]),
        lo_overrun=float(raw_weights["lo_overrun"]),
        hi_overrun=float(raw_weights["hi_overrun"]),
    )
    paper_formula = str(
        payload.get(
            "paper_reward_formula",
            "event_job_start_reward + event_lo_overrun_reward + event_hi_overrun_reward",
        )
    )
    step_formula = str(
        payload.get(
            "step_reward_formula",
            "event_job_start_reward + event_lo_overrun_reward + event_hi_overrun_reward + "
            "(noop_bonus_if_noop if is_explicit_noop_action else 0.0) - "
            "budget_change_penalty * budget_change_norm - budget_drift_penalty * budget_drift_mean",
        )
    )
    raw_parameters = payload.get("reward_parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ValueError(f"Reward config {config_path} field 'reward_parameters' must be an object")
    reward_parameters: dict[str, float] = {
        str(key): float(value) for key, value in raw_parameters.items()
    }
    return RewardModeConfig(
        mode=mode,
        weights=weights,
        paper_reward_formula=paper_formula,
        step_reward_formula=step_formula,
        reward_parameters=reward_parameters,
    )


def evaluate_reward_expression(expression: str, variables: dict[str, float | bool]) -> float:
    """安全计算奖励表达式。

    约束：
    - 只允许常量、变量、四则运算、比较、布尔运算、条件表达式；
    - 禁止函数调用、属性访问、下标访问，避免执行任意代码。
    """

    def _eval(node: ast.AST) -> float | bool:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError(f"Unsupported constant type in reward expression: {type(node.value)}")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable '{node.id}' in reward expression")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -float(operand)
            if isinstance(node.op, ast.UAdd):
                return +float(operand)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            raise ValueError("Unsupported unary operator in reward expression")
        if isinstance(node, ast.BinOp):
            left = float(_eval(node.left))
            right = float(_eval(node.right))
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("Unsupported binary operator in reward expression")
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(_eval(value)) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(_eval(value)) for value in node.values)
            raise ValueError("Unsupported boolean operator in reward expression")
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            current_left = left
            for operator, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if isinstance(operator, ast.Lt):
                    ok = float(current_left) < float(right)
                elif isinstance(operator, ast.LtE):
                    ok = float(current_left) <= float(right)
                elif isinstance(operator, ast.Gt):
                    ok = float(current_left) > float(right)
                elif isinstance(operator, ast.GtE):
                    ok = float(current_left) >= float(right)
                elif isinstance(operator, ast.Eq):
                    ok = current_left == right
                elif isinstance(operator, ast.NotEq):
                    ok = current_left != right
                else:
                    raise ValueError("Unsupported comparison operator in reward expression")
                if not ok:
                    return False
                current_left = right
            return True
        if isinstance(node, ast.IfExp):
            return _eval(node.body) if bool(_eval(node.test)) else _eval(node.orelse)
        raise ValueError(f"Unsupported expression node in reward expression: {type(node).__name__}")

    parsed = ast.parse(expression, mode="eval")
    value = _eval(parsed)
    return float(value)
