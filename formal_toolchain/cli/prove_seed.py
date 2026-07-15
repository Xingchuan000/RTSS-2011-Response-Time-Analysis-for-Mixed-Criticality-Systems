"""顶层入口占位；完整工作流属于 Phase L，本阶段保持明确失败。"""
from ._not_implemented import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
