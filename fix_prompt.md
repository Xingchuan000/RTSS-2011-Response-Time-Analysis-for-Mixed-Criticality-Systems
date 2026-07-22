# 修复提示词：AMC 形式化证明框架 Bug 修复

## 背景
当前代码库在 `formal_toolchain/` 下存在多个相互关联的 bug，导致 4 个测试失败（`tests/formal/` 中 112 passed, 4 failed，全部因为 `certified_envelope is None` 导致 `TypeError`），以及若干架构一致性问题。

以下按修复优先级排列，请依次实施。

---

## 1. G-01: `apply_recovery` 不消费 REC 事件（高优先级）

**文件**: `formal_toolchain/reference/executable_semantics.py`
**行号**: 70-71

**问题**:
```python
def apply_recovery(state: ReferenceState) -> ReferenceState:
    return replace(state, mode="LO")
```
其他所有事件处理函数（`apply_removal`、`apply_deadline_observation`、`apply_arrival_batch`、`apply_mode_switch`、`apply_release`）都调用了 `pop_event(state, event)` 从 frontier 中移除已处理的事件。但 `apply_recovery` 只改了 mode，没有移除 REC 事件，导致 REC 事件永远留在 frontier 中反复被处理。

**修复要求**:
给 `apply_recovery` 增加 `event` 参数，并在函数体内调用 `pop_event(state, event)` 来消费该事件。同时更新 `apply_logical_event` 中调用 `apply_recovery` 的地方（第 124-125 行），将 `event` 参数传进去。

---

## 2. H-04: `build_reference_model_conformance_certificate` 中 `predecessors.items()` 在 None 上崩溃（高优先级）

**文件**: `formal_toolchain/reference/model_conformance.py`
**行号**: 143

**问题**:
```python
def build_reference_model_conformance_certificate(
    *,
    ...
    predecessors: Mapping[str, Mapping[str, Any]] | None = None,
    ...
) -> dict[str, Any]:
    ...
    direct_hashes = {
        obligation_id: cert["artifact_hash"]
        for obligation_id, cert in predecessors.items()   # ← 第 143 行，predecessors 可能是 None
        if isinstance(cert, Mapping) and isinstance(cert.get("artifact_hash"), str)
    }
```
参数 `predecessors` 默认值为 `None`，且唯一调用方（`reference_conformance_checker.py:56-70`）没有传 `predecessors` 参数，只传了 `predecessor_summaries`。因此 `predecessors.items()` 会抛 `AttributeError`。

**修复要求**:
将第 143 行的 `predecessors.items()` 改为 `(predecessors or {}).items()`。

---

## 3. B-03/C-04: 组合上下文管线断裂（高优先级）

涉及两个文件：

### 3a. `formal_checks.py` 中 `build_bundle_context` 传参错误

**文件**: `formal_toolchain/core/formal_checks.py`
**行号**: 321-323

**问题**:
```python
bundle_context = build_bundle_context(
    bridge_context_hash=bridge_context["hash"],   # ← 应该传 composition_context_hash
    target_id=fixture_check.get("target_id"), claim="DEPLOYED_HI_SAFETY")
```

**修复要求**:
1. 在调用 `build_bundle_context` 之前，先调用 `build_composition_context(bridge_context_hash=bridge_context["hash"])` 构建 composition_context
2. 将 `bundle_context` 的调用改为 `composition_context_hash=composition_context["hash"]`
3. 将 `composition_context` 也加入 `contexts` 字典

参考 `contexts.py` 中的函数签名：
```python
def build_composition_context(*, bridge_context_hash: str, **inputs: Any) -> dict[str, Any]:
def build_bundle_context(*, composition_context_hash: str, **inputs: Any) -> dict[str, Any]:
```

### 3b. `contexts.py` 中 Registry 与硬编码上下文映射不一致

**文件**: `formal_toolchain/specs/obligation_registry.json`
**行号**: 2203, 2231

**问题**:
Registry 中 `FINITE_BAD_PREFIX_CONTRADICTION` 和 `FINAL_CLAIM_COMPOSITION` 的 `context_layer` 为 `"reference_context"`，但 `contexts.py:79-80` 映射为 `"composition_context"`。两者必须统一。

**修复要求**:
将 `obligation_registry.json` 中这两个条目的 `context_layer` 改为 `"composition_context"`（与 `contexts.py` 保持一致）。

---

## 4. C-01: 编译器/验证器闭包不一致（中优先级）

**文件**: `formal_toolchain/verifier/aggregator.py`
**行号**: 49-82

**问题**:
验证器的 `claim_dependency_closure` 优先从 `proof_role == "mathematical_root"` 的节点出发遍历依赖，而编译器的 `claim_dependency_closure` 从所有 `gates_claims` 包含 claim 的节点出发。两者闭包大小不一致（~93 vs ~52）。

当前 `FINAL_CLAIM_COMPOSITION` 的 `proof_role` 为 `"mathematical_root"`，所以验证器闭包从它出发。这是设计有意如此（数学根节点聚合所有依赖），但需要确认验证器确实能覆盖所有必要义务。

**修复要求**:
不需要修改代码逻辑。但需要确保 `FINAL_CLAIM_COMPOSITION` 在 registry 中的 `depends_on` 列表完整包含所有直接影响最终 claim 的依赖链，确保验证器闭包不会遗漏关键义务。检查 registry 中 `FINAL_CLAIM_COMPOSITION` 的依赖链是否完整覆盖了所有 `gates_claims: ["DEPLOYED_HI_SAFETY"]` 的条目。

---

## 5. D/E: `certified_envelope` 为 None 导致测试失败（最紧急，当前阻塞）

**文件**: `formal_toolchain/verifier/recompute.py`
**行号**: 206 附近

**问题**:
`certified_envelope` 为 `None`，导致后续迭代时报 `TypeError: 'NoneType' object is not iterable`。

**调查与修复要求**:
1. 追踪 `certified_envelope` 的来源——检查 `envelope_state.certified_envelope` 是如何赋值的
2. 查找在 `recompute.py` 中 `_build_fresh_verifier_state` 或类似函数中 `envelope_state` 的构建逻辑
3. 确认 `certified_envelope` 在验证 pipeline 中何时被赋值，以及为何为 `None`
4. 修复根本原因：可能是 `CERTIFIED_ENVELOPE` 义务的 checker 未正确执行，或 candidate bundle 中缺少 certified_envelope artifact

**相关辅助信息**:
- 第 225 行：`certified_envelope=envelope_state.certified_envelope`
- FreshVerifierState 在第 223-234 行构建
- 4 个测试失败均在 `tests/formal/` 中，因 `certified_envelope` 为 `None` 导致

---

## 6. D: `FreshVerifierState` 中四个字段为 None（中优先级）

**文件**: `formal_toolchain/verifier/recompute.py`
**行号**: 184-191, 207-209, 219-221

**问题**:
```python
concrete_preclosed_engine = None      # 第 184 行
concrete_runtime_snapshot = None      # 第 185 行
reference_preclosed_state = None      # 第 186 行
reference_runtime_snapshot = None     # 第 187 行
```
这些字段在 try 块中被尝试赋值，但 try 块的 except 将其重置为 None（第 207-209, 219-221 行）。这说明构建这些对象的代码抛出了异常。

**修复要求**:
1. 检查 `build_formal_scenario`、`EventRuntimeEngine.build`、`build_formal_runtime_snapshot`、`close_timestamp`、`build_reference_runtime_snapshot` 这些函数的调用参数是否正确
2. 具体原因可能是 3a 修复后（composition_context 管线正确构建）后这些错误会自然消失，因为 certified_envelope 已正确传递

---

## 7. J-02: 理论证明对象路径不存在（低优先级）

**文件**: `formal_toolchain/theory/statements/*.json`

**问题**:
多个 statement JSON 文件中的 `proof_object.path` 指向不存在的文件。例如：
- `FINITE_BAD_PREFIX_CONTRADICTION.json` → `"proofs/FINITE_BAD_PREFIX_CONTRADICTION.thm"`（相对路径，预期在 `theory/statements/proofs/` 下，但目录不存在）
- `FINAL_DEPLOYED_HI_SAFETY_COMPOSITION.json` → `"formal_toolchain/theory/proofs/FINAL_DEPLOYED_HI_SAFETY_COMPOSITION.thm"`（绝对风格路径，对应文件存在）
- `PROTECTED_HI_SAFETY_COROLLARY.json` 完全没有 `proof_object` 字段

**修复要求**:
1. 统一所有 statement 的 `proof_object.path` 路径风格：要么全用相对路径（相对于 statement 文件所在目录），要么全用从仓库根目录出发的路径
2. 参考 `FINAL_DEPLOYED_HI_SAFETY_COMPOSITION.json` 的格式（使用从 repo root 出发的路径 `formal_toolchain/theory/proofs/...`）
3. 检查 `PROTECTED_HI_SAFETY_COROLLARY.json` 是否需要添加 `proof_object`（其 `assurance_level` 为 `"DECLARED_AXIOM_TCB"`，如果是 axiom 可以没有 proof_object，但需要确认设计意图）

---

## 验收标准

修复完成后运行：
```bash
# 运行 formal 测试
python -m pytest tests/formal/ -v 2>&1 | tail -20

# 运行 runtime regression
python -m pytest tests/runtime_regression/ -v 2>&1 | tail -20

# 验证没有语法/类型错误
python -m py_compile formal_toolchain/reference/executable_semantics.py
python -m py_compile formal_toolchain/reference/model_conformance.py
python -m py_compile formal_toolchain/core/formal_checks.py
```

预期结果：`tests/formal/` 中 4 个失败变为通过，runtime regression 保持全部通过。
