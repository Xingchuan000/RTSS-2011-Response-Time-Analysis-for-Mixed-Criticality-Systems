from formal_toolchain.reference.protected_priority_prefix.runtime_schema import build_runtime_schema_certificate, verify_runtime_schema_certificate


def test_runtime_schema_is_source_bound_and_all_obligations_are_checked():
    certificate = build_runtime_schema_certificate()
    assert certificate["schema_version"] == "protected-prefix-runtime-schema-v3"
    assert "pp0_transition_status" in certificate
    assert "legacy_ast_checks" in certificate
    assert len(certificate["legacy_ast_checks"]) >= 10
    assert certificate["source_bindings"] is not None
    verified = verify_runtime_schema_certificate(certificate)
    assert verified["status"] == certificate["status"]
    assert verified["certificate_hash"] == certificate["certificate_hash"]
