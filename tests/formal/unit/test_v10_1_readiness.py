from formal_toolchain.cli.proof_readiness import readiness_report


def test_v10_1_pcssc_readiness_tracks_z3_dependency_only():
    report = readiness_report()
    assert report["proof_route"].endswith("V10_1")
    assert report["framework_revision"] == "V10.17_CRT_PHASE_FAMILY_TERMINAL"
    assert report["proof_pipeline_ready"] is report["formal_dependency_z3_available"]
    assert report["terminal_routes"]["PCSSC"]["event_graph_required"] is False
    assert report["terminal_routes"]["PCSSC"]["canonical_case_consistent_terminal"] is True
    assert report["terminal_routes"]["PCSSC"]["pre_hi_v10_17_mixed_phase_terminals"] is True
    assert report["terminal_routes"]["PCSSC"]["crt_phase_family_terminal"] is True
    assert report["terminal_routes"]["PCSSC"]["lo_entry_v10_13_refinement"] is True
    assert report["terminal_routes"]["BASE_C_AMC_SEM"]["ready"] is True
    assert report["terminal_routes"]["BASE_C_AMC_SEM"]["requires_z3"] is False
