# q-AMC V5 runtime semantics

This repository instantiates q-AMC as a native quality state machine plus an
existing budget-threshold overlay. The native controller owns quality state;
DQN, heuristic, and VIPER actions only update `BudgetState` for future releases.

The four-level profile is synthetic and reproducible because the current
workload exposes scalar full-quality execution samples rather than measured
per-quality CNN WCETs. The profile uses `W/I = 0.5`, integerized by minimum
ratio error, and maps a sample by splitting it into a capped application
component and an observed interference component.

For q-AMC jobs, the release snapshot contains target quality, full sample,
mapped demand, and release-time budget. A later budget update cannot mutate that
snapshot. A quality transition also affects only the next release; it never
rebases or changes the budget state.

The q-AMC event scheduler is local to `RuntimeSemantics.Q_AMC`. If a job needs
exactly one tick beyond its release budget, the overrun event wins over a
same-time completion. A degradable LO task remains in LO mode after its job is
stopped and loses service; an overrun at the minimum quality enters the standard
HI fallback. Quality is persistent across idle mode recovery.

The agent does not observe quality level directly. The q-AMC budget-control
problem is therefore treated as a partially observable problem. The new q-AMC
metrics are independent of the legacy C-AMC-sem `Job.is_degraded` field.

No q-AMC path is connected to the formal proof toolchain, and no q-AMC formal
safety claim is made.

The experiment artifact chain is fail-closed. A q-AMC run binds the frozen
reference configuration, profile manifest, profile spec, taskset profile, DQN
checkpoint, and VIPER tree through canonical fingerprints. A mismatch stops
training or HOUT instead of silently falling back to CLI defaults.

Evaluation method names are:

```text
amc_same_full_sample_native
q_amc_native
q_amc_budget_heuristic
q_amc_dqn_budget_overlay
q_amc_viper_budget_overlay
```

Every q-AMC evaluation row includes q-AMC-specific quality metrics. Legacy
`lo_degraded_*` fields remain for CSV compatibility but are marked inapplicable
to q-AMC. Budget updates are tagged as `DQN_ACTION`, `VIPER_ACTION`,
`HEURISTIC_ACTION`, `OFFLINE_PREQUEUED`, or `UNSPECIFIED`.
