# Archive 4.3 Audit Baseline

Date: 2026-07-22

## Test Results

- formal tests: 111 passed, 6 failed
- runtime tests: 17 passed
- Phase M: dependency lock mismatch

## Registry Closure

- compiler closure: 93
- verifier closure: 52

## Key Deficiencies

1. Source root inferred from request path (not explicit --source-root)
2. Composition context not wired into candidate pipeline
3. Compiler/verifier use different claim closure (93 vs 52)
4. FreshVerifierState has four core fields as None
5. Reference semantics not independently executable
6. N6 bad-prefix builds witness from empty miss ledger
7. Theory proof objects cannot be loaded
8. Schema $ref resolution fails for common_certificate.schema.json

## Acceptance Criteria

1. dependency lock PASS
2. compiler/verifier candidate closure identical
3. component contexts include composition_context
4. All candidate/fresh certificates pass Registry-selected Schema
5. All Registry predecessors exact and PASS
6. Fresh concrete/reference snapshots independently constructed by verifier
7. N4 state relation has no vacuous branches
8. N5 uses parameterized prefix lemma, not fixed job_slots
9. N6 reconstructs from real first-HI-miss set
10. Required theory proof objects loadable and verified or honestly listed in TCB
11. FINAL_CLAIM_COMPOSITION is sole mathematical root
12. All authorization/structural gates PASS
