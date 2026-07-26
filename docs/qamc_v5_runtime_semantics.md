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

## LO release metric population

q-AMC normalized QoS uses every LO task release as its denominator. An LO job
released in HI mode and discarded immediately is a zero-quality release. The
same all-release population is used by `qamc_release_count`, provided-quality
sums, zero-service metrics, unconditional per-release quality, and the q-AMC
loss breakdown.

Release-time target-quality statistics use only managed releases: LO jobs for
which q-AMC created a release-time target snapshot. `qamc_managed_release_count`
reports this population. HI-mode discards do not receive a synthetic target
quality and therefore do not appear in target-rank histograms.

Validation remains fail-closed: generic LO QoS must equal q-AMC normalized QoS
within an absolute tolerance of `1e-12`, q-AMC release count must equal the
independently counted LO releases, and every LO release must appear in exactly
one q-AMC loss bucket.

## Real-seed pilot workflow

Prepare and validate the frozen references and q-AMC profiles without starting
training:

```powershell
.\scripts\run_dqn_amc_family_t10_e1350_h2_h5_perseed.ps1 `
  -RuntimeSemantics Q_AMC `
  -PrepareArtifactsOnly
```

Run the isolated seed185 pilot:

```powershell
.\scripts\run_dqn_amc_family_t10_e1350_h2_h5_perseed.ps1 `
  -RuntimeSemantics Q_AMC `
  -Pilot
```

Pilot artifacts are written below `outputs\dqs_t10_e1350_pilot`, separate from
the full campaign. Run a second real seed with
`-PilotTasksetSeeds 397`.

After training and HOUT complete, apply the readiness gate:

```bash
python -m scripts.check_qamc_pilot_readiness \
  --train-output outputs/dqs_t10_e1350_pilot/tr \
  --hout-output outputs/dqs_t10_e1350_pilot/ho/allbase_qamc_v5
```

The gate writes `pilot_readiness_summary.json` and returns success only when it
prints `READY_FOR_FULL_TRAINING`. Missing checkpoints or validation artifacts,
QoS mismatch, non-conservative loss accounting, absent q-AMC events, absent DQN
budget updates, any HI deadline miss, or incomplete q-AMC native/DQN HOUT all
produce a nonzero exit code.
