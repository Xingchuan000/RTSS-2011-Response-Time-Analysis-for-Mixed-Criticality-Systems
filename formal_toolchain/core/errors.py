"""可分类的工具链错误。"""


class FormalToolchainError(Exception):
    """输入、schema 或 registry 不满足冻结合同。"""

    route = "UNRESOLVED"
    exit_code = 20

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class FormalWorkflowError(FormalToolchainError):
    """可被顶层 workflow 精确路由的合同错误。"""


class ModelConformanceError(FormalWorkflowError):
    route = "MODEL_CONFORMANCE_FAILED"
    exit_code = 10


class BundleInputError(FormalWorkflowError):
    route = "PROOF_BUNDLE_INVALID"
    exit_code = 30


class UnresolvedInputError(FormalWorkflowError):
    route = "UNRESOLVED"
    exit_code = 20
