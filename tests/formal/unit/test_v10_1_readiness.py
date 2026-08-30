from formal_toolchain.cli.proof_readiness import readiness_report


def test_v10_1_pcssc_readiness_tracks_z3_dependency_only():
    report = readiness_report()
    assert report["proof_route"].endswith("V10_1")
    assert report["proof_pipeline_ready"] is report["formal_dependency_z3_available"]
    assert report["terminal_routes"]["PCSSC"]["event_graph_required"] is False
    assert report["terminal_routes"]["BASE_C_AMC_SEM"]["ready"] is False
