# q-AMC V5 implementation and audit report

## Outcome

The q-AMC path now runs end to end as:

```text
profile materialization
  -> native q-AMC / AMC same-sample baselines
  -> DQN-on-q-AMC training
  -> q-AMC HOUT evaluation
  -> VIPER collection, training, and HOUT evaluation
```

The implementation keeps q-AMC quality state in the native runtime and permits
agents to change only release-time budget thresholds. It does not connect
q-AMC to the formal proof route.

## Defects found and fixed

1. q-AMC metrics existed but were never written by the evaluation command.
   Per-seed and unified-summary CSVs now include raw/normalized quality,
   zero-service, degradation, threshold-origin, occupancy, and update-source
   metrics.
2. The required native method matrix was absent. Evaluation now supports
   `amc_same_full_sample_native`, `q_amc_native`,
   `q_amc_budget_heuristic`, `q_amc_dqn_budget_overlay`, and
   `q_amc_viper_budget_overlay`.
3. The training validation baseline selected `Q_AMC` without passing its
   profile bundle and crashed. It now resolves and validates the taskset
   profile before simulation.
4. VIPER collection/training rejected `Q_AMC` at the CLI and did not bind
   artifacts to q-AMC inputs. Both CLIs now accept q-AMC and persist/validate
   reference, manifest, and spec fingerprints.
5. Profile manifest, spec, entry, taskset, and bundle fingerprints were mostly
   trusted rather than checked. The whole artifact chain now fails closed.
6. A frozen reference was checked only for file presence/schema. Runtime now
   verifies its canonical fingerprint, source config, source tree, reward
   artifact, and effective experiment parameters. DQN checkpoints and tree
   artifacts are also bound to these fingerprints.
7. The environment did not use the shared action executor and the wrapper did
   not receive deploy-cap semantics. Canonical static actions now share the
   candidate/rounding/floor/cap/HI-guard/safety pipeline; heuristic and VIPER
   updates have distinct source tags.
8. The fixed q-AMC `W_max` floor was silently disabled when the reference floor
   ratio was zero. It is now always enforced.
9. `qamc_would_overrun_design_c_lo` used executed progress at an early learned
   threshold stop, undercounting design-budget overruns. It now uses the
   release demand.
10. Time-at-quality metrics returned one value of `1.0` per task rather than
    the raw-rank distribution. They now integrate state intervals over
    `N_LO * horizon`, using the configured q-AMC evaluation horizon.
11. A terminal minimum-quality fallback could remain in HI mode forever when
    the triggering job left the processor idle. Idle recovery now runs at that
    boundary without restoring quality.
12. Prequeued budget updates were mislabeled `UNSPECIFIED`; they are now
    `OFFLINE_PREQUEUED`, and q-AMC exports event/task counts for all stable
    sources.
13. Profile/spec validation accepted non-finite ratios and inconsistent
    levels/rules. It now rejects malformed or unsupported artifacts early.

## Verified commands

- Focused q-AMC artifact/runtime tests: passed.
- Changed-path regression suite (runtime, action mask/floor, wrapper, DQN
  factory, train/evaluate CLI, and VIPER metadata/training): 77 passed.
- One-episode DQN-on-q-AMC train plus native/heuristic/DQN HOUT: passed.
- q-AMC teacher collection, one-iteration VIPER training, and VIPER HOUT:
  passed; output source counts distinguish `DQN_ACTION` and `VIPER_ACTION`.

The repository-wide test suite was already red at the audited starting commit:
`942 passed, 1 skipped, 41 failed`. Those failures are primarily in the
separate formal protected-prefix chain and existing training/VIPER tests; they
are not presented as q-AMC proof evidence. q-AMC remains an empirical route.

## Required external artifact

A real successful C-AMC-sem run directory containing `config.json` and its
reward artifact is still required to freeze the production reference. The
full campaign is intentionally not auto-started.
