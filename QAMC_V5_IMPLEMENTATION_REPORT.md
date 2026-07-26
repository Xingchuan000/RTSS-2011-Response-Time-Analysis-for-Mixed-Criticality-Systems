# q-AMC V5 implementation report

Implemented the q-AMC native runtime and the DQN budget-overlay foundation.

Completed components:

- `amc_py/qamc/`: immutable profile/spec models, deterministic integer
  partition, scalar demand mapping, quality controller, scenario adapter, and
  q-AMC metrics;
- q-AMC runtime enum/config validation, release snapshots, strict
  `budget+1` overrun scheduling, next-release-only degradation, terminal HI
  fallback, idle recovery without quality restoration, and source-tagged budget
  updates;
- shared `amc_py/rl/action_execution.py`, used by the runtime wrapper;
- profile spec/config and reference/profile materialization scripts;
- q-AMC CLI choices and required artifact-path validation in train/evaluate
  entry points;
- focused q-AMC boundary tests and runtime semantics documentation.

Validation:

- `pytest -q tests/test_qamc_v5_runtime.py`: passed;
- targeted legacy runtime and DQN factory tests: passed;
- `pytest -q tests --ignore=tests/formal`: the existing repository suite has
  unrelated pre-existing failures; none were in the q-AMC focused tests;
- formal-toolchain code was not changed.

The full q-AMC campaign is not started automatically. It requires a real
reference `config.json`, its frozen artifact, and a per-taskset profile
manifest.
