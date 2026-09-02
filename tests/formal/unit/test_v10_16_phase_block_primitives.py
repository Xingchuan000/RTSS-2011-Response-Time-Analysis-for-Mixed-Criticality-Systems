from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    fixed_phase_post_switch_future_work,
    fixed_phase_pre_hi_carry,
    phase_block_completion_carry_upper,
    phase_block_post_switch_future_upper,
    phase_block_r7_carry_upper,
    phase_block_task_projections,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
)


def _specs():
    return (
        CarryTaskSpec("h", "HI", 10, 2, 5),
        CarryTaskSpec("l", "LO", 15, 4, 1),
    )


def test_v10_16_empty_ahead_joint_period_is_one():
    assert target_release_joint_phase_parameters(20, 5, 0, ()) == (0, 1, 1)


def test_v10_16_symbolic_projection_matches_complete_q_block():
    specs = _specs()
    n0, step, cycle = target_release_joint_phase_parameters(12, 5, 0, specs)
    modulus = 2 if cycle % 2 == 0 else 1
    residue = 0
    projections = phase_block_task_projections(
        12, step, n0, specs, block_modulus=modulus, block_residue=residue
    )
    qs = [q for q in range(cycle) if q % modulus == residue]
    for task_index, projection in enumerate(projections):
        symbolic = {
            projection.phase_residue + k * projection.phase_stride
            for k in range(projection.phase_count)
        }
        explicit = {
            target_release_joint_phases_at_q(
                12, specs, n0=n0, q_step=step, q=q
            )[task_index]
            for q in qs
        }
        assert symbolic == explicit


def test_v10_16_block_lifting_dominates_every_fixed_q_member():
    specs = _specs()
    n0, step, cycle = target_release_joint_phase_parameters(12, 5, 0, specs)
    projections = phase_block_task_projections(
        12, step, n0, specs, block_modulus=1, block_residue=0
    )
    carry_r7, details = phase_block_r7_carry_upper(specs, projections)
    completion_bounds = (8, 10)
    carry_comp = phase_block_completion_carry_upper(
        specs, projections, completion_bounds
    )
    future = phase_block_post_switch_future_upper(specs, projections, 20)

    assert details["candidate_domain_kind"] == "PROVED_BOUNDARY_UNION"
    for q in range(cycle):
        phases = target_release_joint_phases_at_q(
            12, specs, n0=n0, q_step=step, q=q
        )
        fixed_r7, _ = fixed_phase_pre_hi_carry(specs, phases, None)
        _, fixed_with_completion = fixed_phase_pre_hi_carry(
            specs, phases, completion_bounds
        )
        fixed_future = fixed_phase_post_switch_future_work(specs, phases, 20)
        assert carry_r7 >= fixed_r7
        assert carry_comp >= fixed_with_completion["completion_carry_bound"]
        assert future >= fixed_future


def test_v10_16_child_projection_lifting_never_exceeds_root_for_builtins():
    specs = _specs()
    n0, step, cycle = target_release_joint_phase_parameters(12, 5, 0, specs)
    if cycle % 2:
        return
    root = phase_block_task_projections(
        12, step, n0, specs, block_modulus=1, block_residue=0
    )
    root_r7, _ = phase_block_r7_carry_upper(specs, root)
    root_comp = phase_block_completion_carry_upper(specs, root, (8, 10))
    root_future = phase_block_post_switch_future_upper(specs, root, 20)
    for residue in (0, 1):
        child = phase_block_task_projections(
            12, step, n0, specs, block_modulus=2, block_residue=residue
        )
        child_r7, _ = phase_block_r7_carry_upper(specs, child)
        child_comp = phase_block_completion_carry_upper(specs, child, (8, 10))
        child_future = phase_block_post_switch_future_upper(specs, child, 20)
        assert child_r7 <= root_r7
        assert child_comp <= root_comp
        assert child_future <= root_future
