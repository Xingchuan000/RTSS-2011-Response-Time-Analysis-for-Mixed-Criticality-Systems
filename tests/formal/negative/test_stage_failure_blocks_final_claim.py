"""R07：失败路由通过 canonical aggregator 传播到最终 claim。"""

from formal_toolchain.verifier.aggregator import aggregate


def test_stage_failure_blocks_final_claim() -> None:
    assert aggregate(
        [{"id": "A", "status": "PASS"},
         {"id": "B", "status": "UNRESOLVED"}],
        {"A", "B"},
    ) == "UNRESOLVED"
