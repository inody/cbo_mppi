"""Fresh computation of the three manuscript landscape diagnostics."""
from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cbo_mppi.utils import free_energy, temporally_correlated_normal
from experiments.navigation.config import make_problem
from experiments.navigation.env import (
    make_initial_nominal,
    rollout_batch as navigation_rollout,
    straight_line_nominal,
)
from experiments.pendulum_swingup.config import LandscapeConfig, PendulumConfig
from experiments.pendulum_swingup.landscape import build_grid_templates, evaluate_templates
from experiments.planar_arm.closed_loop import run_arm_closed_loop
from experiments.planar_arm.env import (
    end_effector,
    initial_state as arm_initial_state,
    make_arm_hard_rollout,
    trajectory_min_clearance,
)

from .nominal import Cell, arm_problem, controller_config, navigation_raw

ROOT = Path(__file__).resolve().parents[3]


@partial(jax.jit, static_argnames=("clip_inputs",))
def _navigation_landscape_costs(
    x0, center, noise, previous_control, problem, *, clip_inputs
):
    controls = center[None] + noise
    if clip_inputs:
        controls = jnp.clip(controls, -problem.u_max, problem.u_max)
    costs, _ = navigation_rollout(x0, controls, problem, previous_control)
    direct, _ = navigation_rollout(x0, center[None], problem, previous_control)
    return direct[0], costs


def _navigation_antithetic(key, samples, horizon, sigma, correlation):
    base = sigma * temporally_correlated_normal(
        key, (samples // 4, horizon, 2), correlation
    )
    xs = jnp.asarray([-1.0, 1.0])
    ys = jnp.asarray([1.0, -1.0])
    return jnp.concatenate((base, base * xs, base * ys, -base), axis=0)


def navigation_routes(output_root: Path) -> Path:
    destination = output_root / "landscapes/navigation/route_heights.npz"
    if destination.exists():
        raise FileExistsError(destination)
    raw = navigation_raw(Cell("navigation", "mppi", 15.0))
    problem = make_problem(raw)
    x0 = jnp.asarray(raw["problem"]["initial_state"], dtype=jnp.float32)
    h_values = np.asarray([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], dtype=np.float32)
    trajectories = []
    for height in h_values:
        center = straight_line_nominal(
            x0, problem, int(raw["problem"]["horizon"]), route_height=float(height)
        )
        _, trajectory = navigation_rollout(
            x0, center[None], problem, jnp.zeros(2), clip_inputs=True
        )
        trajectories.append(np.asarray(trajectory[0, :, :2]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        h_values=h_values,
        trajectories=np.asarray(trajectories),
        obstacle_centers=np.asarray(problem.obstacle_centers),
        obstacle_radii=np.asarray(problem.obstacle_radii),
        start=np.asarray(x0[:2]),
        goal=np.asarray(problem.goal),
    )
    return destination


def navigation_landscape(output_root: Path) -> Path:
    destination = output_root / "landscapes/navigation/free_energy_landscape.npz"
    if destination.exists():
        raise FileExistsError(destination)
    horizon, samples = 60, 65_536
    raw = navigation_raw(Cell("navigation", "mppi", 15.0))
    raw["problem"]["horizon"] = horizon
    raw["problem"]["obstacle_radii"] = [0.5, 0.5]
    problem = make_problem(raw)
    x0 = jnp.asarray(raw["problem"]["initial_state"], dtype=jnp.float32)
    previous_control = jnp.zeros(2, dtype=jnp.float32)
    profile = str(raw["problem"].get("nominal_profile", "minimum_jerk"))
    noise = _navigation_antithetic(
        jax.random.PRNGKey(20_260_621), samples, horizon, 0.35, 0.9
    )
    heights = np.linspace(-3.0, 3.0, 121)
    nominal_costs = []
    taus = np.asarray([15.0, 300.0])
    free_energies = {float(tau): [] for tau in taus}
    for height in heights:
        center = make_initial_nominal(x0, problem, horizon, float(height), profile)
        direct, costs = jax.device_get(
            _navigation_landscape_costs(
                x0, center, noise, previous_control, problem, clip_inputs=True
            )
        )
        nominal_costs.append(float(direct))
        costs = jnp.asarray(np.asarray(costs))
        for tau in taus:
            free_energies[float(tau)].append(float(free_energy(costs, float(tau))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        heights=heights,
        nominal_costs=np.asarray(nominal_costs),
        taus=taus,
        free_energies=np.asarray([free_energies[float(tau)] for tau in taus]),
        horizon=np.asarray(horizon),
        samples=np.asarray(samples),
        seed=np.asarray(20_260_621),
        noise_sigma=np.asarray(0.35),
        noise_correlation=np.asarray(0.9),
        radius=np.asarray(0.5),
        clip_input_samples=np.asarray(True),
    )
    return destination


def pendulum_landscape(output_root: Path) -> Path:
    destination = output_root / "landscapes/pendulum/landscape_tau10.npz"
    if destination.exists():
        raise FileExistsError(destination)
    cfg = PendulumConfig(horizon=160, u_max=3.0)
    lcfg = LandscapeConfig(
        tau=10.0, sigma_local=0.25, rho=0.9, samples=4096,
        grid_n=81, mode="full_double", k1=3.0, k2=4.0,
        target_batch=131_072, seed=27_200,
    )
    templates, _, _, _, basis = build_grid_templates(cfg, lcfg)
    result = evaluate_templates(templates, cfg, lcfg, basis=basis)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        a_vals=result.a_vals,
        b_vals=result.b_vals,
        params=result.params,
        nom_cost=result.nom_cost,
        nom_success=result.nom_success,
        nom_terminal_angle=result.nom_terminal_angle,
        F_hat=result.F_hat,
        pert_success=result.pert_success,
        ess=result.ess,
        mean_cost=result.mean_cost,
        q10_cost=result.q10_cost,
        sigma_local=result.sigma_local,
        rho=result.rho,
        samples=result.samples,
        mode=result.mode,
    )
    return destination


_ARM_ROLLOUT = jax.jit(make_arm_hard_rollout(), static_argnames=("clip_inputs",))


def _arm_jacobian(q: np.ndarray, lengths: tuple[float, float, float]) -> np.ndarray:
    l1, l2, l3 = lengths
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    c123, s123 = np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)
    return np.asarray([
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
    ], dtype=np.float64)


def _raw_null_direction(q: np.ndarray, lengths) -> np.ndarray:
    jacobian = _arm_jacobian(q, lengths)
    raw = np.cross(jacobian[0], jacobian[1])
    norm = float(np.linalg.norm(raw))
    if norm < 1e-10:
        raise RuntimeError("unstable planar-arm null direction")
    return (raw / norm).astype(np.float32)


def _finite_free_energy(costs: np.ndarray, tau: float) -> float:
    finite = np.asarray(costs[np.isfinite(costs)], dtype=np.float64)
    if not finite.size:
        return float("inf")
    minimum = float(np.min(finite))
    scaled_sum = float(np.sum(np.exp(-(finite - minimum) / tau)))
    return float(minimum - tau * np.log(scaled_sum / costs.size))


def _representatives(hs: np.ndarray, nominal_cost: np.ndarray, free_energy_values: np.ndarray) -> np.ndarray:
    def finite_argmin(values):
        return int(np.argmin(np.where(np.isfinite(values), values, np.inf)))
    anchors = [
        int(np.argmin(np.abs(hs + 0.2))),
        int(np.argmin(np.abs(hs - 0.2))),
        int(np.argmin(np.abs(hs))),
        finite_argmin(nominal_cost),
        finite_argmin(free_energy_values[:, 0]),
        finite_argmin(free_energy_values[:, 1]),
    ]
    used: set[int] = set()
    selected = []
    for requested in anchors:
        index = requested if requested not in used else min(
            (i for i in range(len(hs)) if i not in used),
            key=lambda i: (abs(i - requested), i),
        )
        used.add(index)
        selected.append(index)
    return np.asarray(sorted(selected, key=lambda i: (float(hs[i]), i)), dtype=np.int32)


def planar_arm_landscape(output_root: Path) -> tuple[Path, Path]:
    root = output_root / "landscapes/planar_arm"
    arrays_path = root / "landscape_arrays.npz"
    base_path = root / "base_trajectory.npz"
    if arrays_path.exists() or base_path.exists():
        raise FileExistsError(root)
    controller_problem = arm_problem()
    base_result = run_arm_closed_loop(
        "proposed", controller_problem,
        controller_config(Cell("planar_arm", "cbo_mppi", 3.0)),
        seed=60_000, steps=60, process_noise_sigma=0.0,
        hard_obstacles=True, episodic=False,
    )
    base_controls = np.asarray(base_result.controls, dtype=np.float32)
    base_states = np.asarray(base_result.states, dtype=np.float32)
    if base_controls.shape != (60, 3) or base_states.shape != (61, 6):
        raise RuntimeError("unexpected planar-arm base trajectory shape")
    if not np.all(np.asarray(base_result.rollout_counts) == 131_072):
        raise RuntimeError("planar-arm base rollout budget mismatch")
    cfg = controller_problem._replace(horizon=60)
    x0 = arm_initial_state(cfg)
    base_cost, base_trajectory_jax = _ARM_ROLLOUT(
        x0, jnp.asarray(base_controls[None]), cfg, jnp.zeros(3), clip_inputs=True
    )
    base_trajectory = np.asarray(base_trajectory_jax, dtype=np.float32)[0]
    if not np.array_equal(base_trajectory, base_states):
        raise RuntimeError("planar-arm base open-loop replay differs from closed loop")
    null_direction = _raw_null_direction(base_trajectory[-1, :3], cfg.link_lengths)
    hs = np.linspace(-0.3, 0.3, 61, dtype=np.float32)
    s = np.arange(60, dtype=np.float32) / 59.0
    ramp = s * s * (3.0 - 2.0 * s)
    family_controls = np.clip(
        base_controls[None] + hs[:, None, None] * 2.0
        * ramp[None, :, None] * null_direction[None, None, :],
        -cfg.u_max, cfg.u_max,
    ).astype(np.float32)
    nominal_cost_jax, trajectories_jax = _ARM_ROLLOUT(
        x0, jnp.asarray(family_controls), cfg, jnp.zeros(3), clip_inputs=True
    )
    nominal_cost = np.asarray(nominal_cost_jax, dtype=np.float64)
    trajectories = np.asarray(trajectories_jax, dtype=np.float32)
    clearances = np.asarray(
        trajectory_min_clearance(jnp.asarray(trajectories), cfg), dtype=np.float64
    )
    final_q = trajectories[:, -1, :3]
    final_ee = np.asarray(end_effector(jnp.asarray(final_q), cfg), dtype=np.float64)
    ee_errors = np.linalg.norm(final_ee - np.asarray(cfg.goal)[None], axis=1)
    noise = 0.03 * temporally_correlated_normal(
        jax.random.PRNGKey(27_200), (1024, 60, 3), 0.81
    )
    perturbed = family_controls[:, None] + np.asarray(noise, dtype=np.float32)[None]
    mc_costs_jax, _ = _ARM_ROLLOUT(
        x0, jnp.asarray(perturbed.reshape(61 * 1024, 60, 3)),
        cfg, jnp.zeros(3), clip_inputs=True,
    )
    mc_costs = np.asarray(mc_costs_jax, dtype=np.float32).reshape(61, 1024)
    taus = np.asarray([3.0, 300.0], dtype=np.float64)
    projected = np.asarray([
        [_finite_free_energy(mc_costs[i], tau) for tau in taus]
        for i in range(len(hs))
    ])
    representatives = _representatives(hs, nominal_cost, projected)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base_path,
        states=np.asarray(base_result.states),
        controls=np.asarray(base_result.controls),
        rollout_counts=np.asarray(base_result.rollout_counts),
        **{f"diagnostic_{name}": np.asarray(value) for name, value in base_result.diagnostics.items()},
    )
    np.savez_compressed(
        arrays_path,
        h=hs,
        base_controls=base_controls,
        base_states=np.asarray(base_result.states, dtype=np.float32),
        base_open_loop_cost=np.asarray(base_cost, dtype=np.float32),
        ramp=ramp,
        null_direction=null_direction,
        family_controls=family_controls,
        family_trajectories=trajectories,
        nominal_cost=nominal_cost,
        projected_free_energy=projected,
        monte_carlo_costs=mc_costs,
        finite_sample_count=np.isfinite(mc_costs).sum(axis=1).astype(np.int32),
        minimum_clearance=clearances,
        final_q=final_q,
        final_end_effector=final_ee,
        final_end_effector_error=ee_errors,
        taus=taus,
        representative_display_indices=representatives,
    )
    return arrays_path, base_path


def run_all(output_root: Path, resume: bool = False) -> list[Path]:
    jobs = [
        ("navigation-routes", navigation_routes, output_root / "landscapes/navigation/route_heights.npz"),
        ("navigation-landscape", navigation_landscape, output_root / "landscapes/navigation/free_energy_landscape.npz"),
        ("pendulum-landscape", pendulum_landscape, output_root / "landscapes/pendulum/landscape_tau10.npz"),
        ("planar-arm-landscape", planar_arm_landscape, output_root / "landscapes/planar_arm/landscape_arrays.npz"),
    ]
    outputs: list[Path] = []
    for name, function, marker in jobs:
        if resume and marker.is_file():
            outputs.append(marker)
            continue
        print(f"[landscapes] {name}", flush=True)
        value = function(output_root)
        outputs.extend(value if isinstance(value, tuple) else (value,))
    manifest = output_root / "landscapes/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "timing": "not measured",
        "files": [str(path.relative_to(output_root)) for path in outputs],
    }, indent=2, sort_keys=True) + "\n")
    return outputs
