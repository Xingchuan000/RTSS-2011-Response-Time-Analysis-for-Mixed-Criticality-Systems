from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path

CODE_ROOT = Path('/mnt/data/review_archive6').resolve()
SOURCE_SEED = Path('/mnt/data/s185_extracted').resolve()
OUT_ROOT = Path('/mnt/data/s185_rebuilt').resolve()
VARIANT = 'best_overall'
SEED = 185

sys.path.insert(0, str(CODE_ROOT))

from formal_toolchain.adapters.amc_taskset import (
    derive_action_task_order,
    derive_feature_task_order,
    export_taskset,
)
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.adapters.s185_target import build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.core.hashing import sha256_file, sha256_object


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
shutil.copytree(SOURCE_SEED, OUT_ROOT)

manifest = read(SOURCE_SEED / 'dataset_train_s1350_1399' / 'manifest.json')
artifact_dir = OUT_ROOT / VARIANT
artifact_features = read(artifact_dir / 'feature_names.json')
artifact_actions = read(artifact_dir / 'action_definitions.json')

w = manifest['workload_cli_config']
r = manifest['runtime_args']
f = manifest['feature_config']

workload_args = {
    'mode': w['mc_fairgen_mode'],
    'num_tasks': int(w['mc_fairgen_num_tasks']),
    'hi_ratio': float(w['mc_fairgen_hi_ratio']),
    'period_source': w['mc_fairgen_period_source'],
    'period_scale': int(w['mc_fairgen_period_scale']),
    'tick_ns': 10,
    'require_schedulable': bool(w['require_schedulable']),
    'check_safety': True,
    'max_attempts': 100,
    'scenario_seed_offset': int(w['scenario_seed_offset']),
    'fixed_taskset_seed': int(w['fixed_taskset_seed']),
    'u_hi_lo_min': float(w['mc_fairgen_u_hi_lo_min']),
    'u_hi_lo_max': float(w['mc_fairgen_u_hi_lo_max']),
    'u_hi_hi_min': float(w['mc_fairgen_u_hi_hi_min']),
    'u_hi_hi_max': float(w['mc_fairgen_u_hi_hi_max']),
    'u_lo_lo_min': float(w['mc_fairgen_u_lo_lo_min']),
    'u_lo_lo_max': float(w['mc_fairgen_u_lo_lo_max']),
    'hi_budget_rho_min': float(w['mc_fairgen_hi_budget_rho_min']),
    'hi_budget_rho_max': float(w['mc_fairgen_hi_budget_rho_max']),
    'lo_budget_rho_min': float(w['mc_fairgen_lo_budget_rho_min']),
    'lo_budget_rho_max': float(w['mc_fairgen_lo_budget_rho_max']),
    'hi_overrun_prob': float(w['mc_fairgen_hi_overrun_prob']),
    'lo_overrun_prob': float(w['mc_fairgen_lo_overrun_prob']),
    'hi_overrun_factor_min': float(w['mc_fairgen_hi_overrun_factor_min']),
    'hi_overrun_factor_max': float(w['mc_fairgen_hi_overrun_factor_max']),
    'lo_overrun_factor_min': float(w['mc_fairgen_lo_overrun_factor_min']),
    'lo_overrun_factor_max': float(w['mc_fairgen_lo_overrun_factor_max']),
}

runtime_args = {
    'runtime_semantics': r['runtime_semantics'],
    'c_amc_sem_xf': float(r['c_amc_sem_xf']),
    'end_time': int(manifest['horizon']),
    'agent_period': int(manifest['agent_period']),
    'action_space': r['action_space'],
    'budget_increase_ratio': float(r['budget_increase_ratio']),
    'budget_decrease_ratio': float(r['budget_decrease_ratio']),
    'include_explicit_noop': bool(r.get('include_explicit_noop', False)),
    'budget_floor_ratio': float(r['budget_floor_ratio']),
    'forbid_decreasing_hi_budgets': bool(r['forbid_decreasing_hi_budgets']),
    'mask_detail_mode': r['mask_detail_mode'],
    'enable_deploy_cap_mask': bool(r['enable_deploy_cap_mask']),
    'deploy_cap_mask_ratio': float(r['deploy_cap_mask_ratio']),
    'deploy_cap_mask_criticality': r['deploy_cap_mask_criticality'],
    'capture_trace': True,
    'capture_debug_events': False,
    'processor_overhead': 0,
}

recipe = {
    'schema_version': 'real_viper_seed_target_recipe_v1',
    'factory': 'formal_toolchain.adapters.s185_target:build_target',
    'kwargs': {
        'seed': SEED,
        'workload_args': workload_args,
        'runtime_args': runtime_args,
        'feature_config': dict(f),
        'expected_feature_names': artifact_features,
        'expected_action_definitions': artifact_actions,
        'original_reward_mode': r.get('reward_mode'),
        # The historical reward-mode alias is absent in the current code. Reward
        # does not enter the P0 formal target config or action/feature semantics.
        'formal_reward_mode': 'mendes',
    },
    'uses_real_seed': True,
    'source_dataset_manifest_sha256': sha256_file(SOURCE_SEED / 'dataset_train_s1350_1399' / 'manifest.json'),
    'source_extraction_script_sha256': sha256_file(Path('/mnt/data/run_viper_formalv1_csem_t10_compact_balanced_h2_h5_parallel3_v3.ps1')),
}

# Build through the exact recipe that the proof request will use.
target = build_target(**recipe['kwargs'])
budget = target.provenance['budget_by_task']
taskset = export_taskset(target.ordered_tasks, budget)
effective_config = export_formal_target_config(target)
if effective_config.get('status') != 'PASS':
    raise RuntimeError(f'effective config export failed: {effective_config}')

feature_order = derive_feature_task_order(target.feature_names)
action_order = derive_action_task_order(target.action_definitions)
task_order = [task.name for task in target.ordered_tasks]
if task_order != feature_order or task_order != action_order:
    raise RuntimeError('task/feature/action order mismatch')

inputs = OUT_ROOT / 'formal_inputs'
inputs.mkdir(parents=True, exist_ok=True)
write(inputs / 'target_recipe.json', recipe)
write(inputs / 'code_taskset_canonical.json', taskset)
write(inputs / 'priority_order.json', {
    'schema_version': 'priority_order_v1',
    'priority_order': taskset['priority_order'],
})
write(inputs / 'effective_runtime_config.json', effective_config)
write(inputs / 'action_definitions_canonical.json', {
    'schema_version': 'action_definitions_canonical_v1',
    'action_definitions': list(target.action_definitions),
})
write(inputs / 'feature_schema_canonical.json', {
    'schema_version': 'feature_schema_canonical_v1',
    'feature_names': list(target.feature_names),
    'state_dim': len(target.feature_names),
    'task_feature_order': feature_order,
})

source_manifest = build_source_manifest(CODE_ROOT)
write(inputs / 'source_tree_manifest.json', source_manifest)
write(inputs / 'runtime_environment_manifest.json', {
    'schema_version': 'runtime_environment_manifest_v1',
    'python_version': platform.python_version(),
    'python_implementation': platform.python_implementation(),
    'platform': platform.platform(),
    'byteorder': sys.byteorder,
})
write(inputs / 'dependency_manifest.json', {
    'schema_version': 'dependency_manifest_v1',
    'packages': {
        'numpy': package_version('numpy'),
        'scipy': package_version('scipy'),
        'scikit-learn': package_version('scikit-learn'),
        'torch': package_version('torch'),
        'z3-solver': package_version('z3-solver'),
    },
    'requirements_sha256': sha256_file(CODE_ROOT / 'requirements.txt'),
    'pyproject_sha256': sha256_file(CODE_ROOT / 'pyproject.toml'),
})

variant_hashes = {}
for variant in ('best_overall', 'best_balanced', 'best_performance'):
    directory = OUT_ROOT / variant
    if directory.is_dir():
        variant_hashes[variant] = {
            name: sha256_file(directory / name)
            for name in ('integer_tree.json', 'feature_names.json', 'action_definitions.json', 'fixed_point_config.json', 'metadata.json', 'artifact_manifest.json')
        }

write(inputs / 'provenance.json', {
    'schema_version': 'real_seed_provenance_v2',
    'seed': SEED,
    'taskset_seed': int(target.provenance['taskset_seed']),
    'scenario_seed': int(target.provenance['scenario_seed']),
    'taskset_attempts': int(target.provenance['taskset_attempts']),
    'taskset_fingerprint': taskset['fingerprint'],
    'taskset_fingerprint_short': target.provenance['taskset_fingerprint_short'],
    'tree_variant_exported': VARIANT,
    'variant_file_hashes': variant_hashes,
    'dataset_manifest_sha256': recipe['source_dataset_manifest_sha256'],
    'extraction_script_sha256': recipe['source_extraction_script_sha256'],
    'source_semantic_hash': source_manifest['semantic_hash'],
    'original_reward_mode': r.get('reward_mode'),
    'formal_reward_mode': 'mendes',
    'reward_mode_note': 'The historical alias is unavailable in the current source; reward is excluded from the P0 formal target configuration and does not affect reproduced task/feature/action identity.',
})

# Copy the actual factory source into formal_inputs as frozen provenance. The
# proof process imports the same module from code_root, not this copied file.
shutil.copy2(CODE_ROOT / 'formal_toolchain' / 'adapters' / 's185_target.py', inputs / 'target.py')

write(OUT_ROOT / 'formal_target_manifest.json', {
    'schema_version': 'formal_target_manifest_v1',
    'target_id': 's185',
    'target_kind': 'REAL_VIPER_SEED',
    'taskset_seed': SEED,
    'tree_variants': ['best_overall', 'best_balanced', 'best_performance'],
    'authoritative_input_mode': 'FROZEN_FORMAL_INPUTS',
    'formal_inputs_version': 's185_p0_rebuilt_v1',
})

# Generate a Phase-K path map against the exact current code root.
import subprocess
subprocess.run([
    sys.executable,
    str(CODE_ROOT / 'scripts' / 'regenerate_phase_k_case_map.py'),
    '--out', str(OUT_ROOT / 'phase_k_case_map.json'),
], cwd=CODE_ROOT, check=True, env={**__import__('os').environ, 'PYTHONPATH': str(CODE_ROOT)})

# Inspection.
inventory = inspect_tree_artifact(artifact_dir, expected_state_dim=128, expected_action_dim=24, expected_seed=185)
checks = {
    'target_factory_constructed': True,
    'task_count': len(task_order),
    'state_dim': len(target.feature_names),
    'action_dim': len(target.action_definitions),
    'task_order_equals_feature_order': task_order == feature_order,
    'task_order_equals_action_order': task_order == action_order,
    'artifact_feature_names_match_target': inventory['feature_names'] == list(target.feature_names),
    'artifact_action_definitions_match_target': inventory['action_definitions'] == list(target.action_definitions),
    'artifact_seed_matches': inventory['metadata_taskset_seed'] == 185,
    'fixed_point_config_hash': inventory['fixed_point_config_hash'],
    'taskset_fingerprint': taskset['fingerprint'],
    'source_semantic_hash': source_manifest['semantic_hash'],
    'effective_runtime_config_status': effective_config['status'],
    'priority_order': task_order,
}
write(inputs / 'rebuild_check.json', checks)

report = f'''# s185 formal_inputs 重建检查报告\n\n## 结论\n\n已从现有 `s185` 数据集 manifest、整数树 artifact、VIPER 提取脚本和当前代码重建 `formal_inputs`，未重新训练 DQN，也未重新运行 VIPER 提取。\n\n## 已通过检查\n\n- 任务数：{len(task_order)}\n- 特征维度：{len(target.feature_names)}\n- 动作维度：{len(target.action_definitions)}\n- task/feature 顺序一致：{task_order == feature_order}\n- task/action 顺序一致：{task_order == action_order}\n- artifact feature 与 target 一致：{inventory['feature_names'] == list(target.feature_names)}\n- artifact action 与 target 一致：{inventory['action_definitions'] == list(target.action_definitions)}\n- artifact taskset seed：{inventory['metadata_taskset_seed']}\n- fixed-point config hash：`{inventory['fixed_point_config_hash']}`\n- canonical taskset fingerprint：`{taskset['fingerprint']}`\n- current source semantic hash：`{source_manifest['semantic_hash']}`\n\n## 权威优先级顺序\n\n```text\n{', '.join(task_order)}\n```\n\n## 注意事项\n\n训练 manifest 中的 reward mode `{r.get('reward_mode')}` 在当前代码中已不存在。重建 target 使用当前可用的 `mendes` 作为环境构造占位；reward mode 不进入 P0 formal target 配置，并且验证表明任务集、128 个 feature 与 24 个 action 均与原 artifact 完全一致。\n\n`phase_k_case_map.json` 已按当前代码重新生成，因此它只适用于当前归档源码；以后修改被绑定的运行时代码后需要重新生成。\n'''
(OUT_ROOT / 'formal_inputs重建检查报告.md').write_text(report, encoding='utf-8')

zip_path = Path('/mnt/data/s185_with_rebuilt_formal_inputs.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUT_ROOT.rglob('*'):
        if path.is_file():
            zf.write(path, Path('s185') / path.relative_to(OUT_ROOT))

patch_root = Path('/mnt/data/s185_target_factory_patch')
if patch_root.exists():
    shutil.rmtree(patch_root)
(patch_root / 'formal_toolchain' / 'adapters').mkdir(parents=True)
shutil.copy2(CODE_ROOT / 'formal_toolchain' / 'adapters' / 's185_target.py', patch_root / 'formal_toolchain' / 'adapters' / 's185_target.py')
shutil.copy2(Path('/mnt/data/rebuild_s185_formal_inputs.py'), patch_root / 'rebuild_s185_formal_inputs.py')

print(json.dumps({'out_root': str(OUT_ROOT), 'zip': str(zip_path), 'checks': checks}, ensure_ascii=False, indent=2))
