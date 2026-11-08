from __future__ import annotations

"""Compact, matched comparison of the current center restraint and a 3-2-1 mount.

Both cases use the same field-cooling-fixed flux-pinning matrices, mechanism
damping ratio, impulse, segment properties, geometry, and integration time.
The center case uses 500 N/m and 500 N m/rad with 1% damping. The 3-2-1
modal result is enforced exactly by null-space reduction; its transient uses
a finite point-spring approximation with zero support damping by default,
consistent with treating the literature kinematic mount as a boundary
constraint rather than a measured dissipative element. The exact and penalty
modal frequencies are both reported so the
quality of that transient approximation is visible.
"""

import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from scipy.linalg import eigh, null_space

import segment_dynamics as sim
from vertical_placement_study import adjacent_pairs, build_linear_system
from mechanism_parameters import CONFIG, mechanism_C6, mechanism_K6, temperature_ratio
from empirical_mechanism_data import (
    CHIANG_FORCE_N, CHIANG_T_K, JIANG_FORCE_N, JIANG_T_K,
    jiang_pchip_force_n,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Support_Boundary_Comparison_Output"
Z_MM = 22.5
HALF_SPAN_MM = 225.0
MECHANISM_ZETA = 0.03
CENTER_RESTRAINT_ZETA = 0.01
DEFAULT_KINEMATIC_SUPPORT_ZETA = 0.0
SUPPORT_SEGMENTS = (1, 2, 6)

# Fixed N=4 isolates support topology.  The final two cases show the
# model-form spread between the independent temperature datasets.
CASES = (
    ("chiang_stiff", "chiang", 50.0, 7.5, 4),
    ("reference", "chiang", 77.0, 10.0, 4),
    ("chiang_compliant", "chiang", 85.0, 25.0, 4),
    ("jiang_compliant", "jiang", 85.0, 25.0, 4),
)


@dataclass(frozen=True)
class Support:
    name: str
    segment: int
    local_point: np.ndarray
    axes: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--run-time", type=float, default=16.0)
    parser.add_argument("--timestep", type=float, default=1.0e-4)
    parser.add_argument("--support-stiffness", type=float, default=1.0e4)
    parser.add_argument(
        "--kinematic-support-zeta", type=float,
        default=DEFAULT_KINEMATIC_SUPPORT_ZETA,
        help="Damping ratio assigned to finite 3-2-1 penalty supports. Default 0; use 0.01 only for a matched-damping sensitivity run.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = map(float, v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def positions(separation_mm: float) -> np.ndarray:
    return np.asarray(sim.hex_positions(1, sim.GEOM.hex_radius, separation_mm * 1e-3))


def supports(separation_mm: float) -> list[Support]:
    pos = positions(separation_mm)
    apothem = math.sqrt(3.0) * 0.5 * sim.GEOM.hex_radius

    def radial_point(index: int) -> np.ndarray:
        direction = pos[index] / np.linalg.norm(pos[index])
        return np.array([apothem * direction[0], apothem * direction[1], 0.0])

    ex, ey, ez = np.eye(3)
    return [
        Support("A: 3 DOF", 1, radial_point(1), (ex, ey, ez)),
        Support("B: 2 DOF", 2, radial_point(2), (ey, ez)),
        Support("C: 1 DOF", 6, radial_point(6), (ez,)),
    ]


def constraint_matrix(items: list[Support], nbody: int = 7) -> np.ndarray:
    rows = []
    for item in items:
        point_map = np.block([np.eye(3), -skew(item.local_point)])
        for axis in item.axes:
            row = np.zeros(6 * nbody)
            row[6 * item.segment:6 * item.segment + 6] = axis @ point_map
            rows.append(row)
    H = np.asarray(rows)
    if H.shape != (6, 6 * nbody) or np.linalg.matrix_rank(H) != 6:
        raise RuntimeError("3-2-1 constraint matrix does not have rank six.")
    return H


@contextmanager
def model_settings(*, center_enabled: bool):
    old = (
        sim.DYN.anchor_k_trans, sim.DYN.anchor_k_rot,
        sim.DYN.anchor_zeta_trans, sim.DYN.anchor_zeta_rot,
        sim.DYN.zeta_nominal,
    )
    sim.DYN.anchor_k_trans = 500.0 if center_enabled else 0.0
    sim.DYN.anchor_k_rot = 500.0 if center_enabled else 0.0
    sim.DYN.anchor_zeta_trans = CENTER_RESTRAINT_ZETA if center_enabled else 0.0
    sim.DYN.anchor_zeta_rot = CENTER_RESTRAINT_ZETA if center_enabled else 0.0
    sim.DYN.zeta_nominal = MECHANISM_ZETA
    try:
        yield
    finally:
        (
            sim.DYN.anchor_k_trans, sim.DYN.anchor_k_rot,
            sim.DYN.anchor_zeta_trans, sim.DYN.anchor_zeta_rot,
            sim.DYN.zeta_nominal,
        ) = old


def first_mode(K: np.ndarray, C: np.ndarray, M: np.ndarray) -> dict:
    values, vectors = eigh(K, M)
    frequencies = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    valid = np.flatnonzero(frequencies > 1e-7)
    i = int(valid[0])
    phi = vectors[:, i]
    omega = 2.0 * np.pi * frequencies[i]
    modal_mass = float(phi @ M @ phi)
    modal_damping = float(phi @ C @ phi)
    return {
        "frequency_hz": float(frequencies[i]),
        "modal_damping_ratio": float(modal_damping / (2.0 * omega * modal_mass)),
        "first_six_positive_frequencies_hz": frequencies[valid[:6]].tolist(),
    }


def modal_results(model: str, temperature: float, separation: float, count: int,
                  support_stiffness: float, kinematic_support_zeta: float) -> dict:
    with model_settings(center_enabled=True):
        Kc, Cc, Mc = build_linear_system(
            temperature, model, count, separation, Z_MM, HALF_SPAN_MM
        )
    center = first_mode(Kc, Cc, Mc)

    with model_settings(center_enabled=False):
        K, C, M = build_linear_system(
            temperature, model, count, separation, Z_MM, HALF_SPAN_MM
        )
    H = constraint_matrix(supports(separation), M.shape[0] // 6)
    Z = null_space(H)
    exact = first_mode(Z.T @ K @ Z, Z.T @ C @ Z, Z.T @ M @ Z)

    c_support = 2.0 * kinematic_support_zeta * math.sqrt(
        support_stiffness * CONFIG.segment.mass_kg
    )
    Kp = K + support_stiffness * (H.T @ H)
    Cp = C + c_support * (H.T @ H)
    penalty = first_mode(Kp, Cp, M)
    return {"center": center, "kinematic_exact": exact, "kinematic_penalty": penalty}


def make_penalty_support(separation_mm: float, stiffness: float, support_zeta: float):
    items = supports(separation_mm)
    pos = positions(separation_mm)
    refs = {
        item.segment: np.r_[pos[item.segment], sim.GEOM.body_z] + item.local_point
        for item in items
    }
    damping = 2.0 * support_zeta * math.sqrt(
        stiffness * CONFIG.segment.mass_kg
    )
    state = {"max_displacement_m": 0.0, "max_force_N": 0.0}

    def apply(model, data, _body_id, _p_ref, _R_ref):
        for item in items:
            body_id = model.body(f"h{item.segment}").id
            center, rotation, velocity, omega = sim.body_pose(model, data, body_id)
            arm = rotation @ item.local_point
            point = center + arm
            point_velocity = velocity + np.cross(omega, arm)
            displacement = point - refs[item.segment]
            force = np.zeros(3)
            for axis in item.axes:
                x = float(axis @ displacement)
                v = float(axis @ point_velocity)
                force += (-stiffness * x - damping * v) * axis
                state["max_displacement_m"] = max(state["max_displacement_m"], abs(x))
            state["max_force_N"] = max(state["max_force_N"], float(np.linalg.norm(force)))
            sim.apply_point_force(model, data, body_id, point, force)

    return apply, state


@contextmanager
def transient_boundary(boundary: str, separation_mm: float, stiffness: float,
                       kinematic_support_zeta: float):
    old_apply = sim.apply_anchor
    if boundary == "center_1pct":
        with model_settings(center_enabled=True):
            yield {"max_displacement_m": 0.0, "max_force_N": 0.0}
        return
    apply, state = make_penalty_support(
        separation_mm, stiffness, kinematic_support_zeta
    )
    with model_settings(center_enabled=False):
        sim.apply_anchor = apply
        try:
            yield state
        finally:
            sim.apply_anchor = old_apply


def response_histories(result, separation_mm: float) -> dict[str, np.ndarray]:
    t = np.asarray(result.t)
    dofs = np.asarray(result.segment_dofs)
    rel_pos = np.zeros(len(t))
    rel_rot = np.zeros(len(t))
    for i, j in adjacent_pairs(
        sim.hex_positions(1, sim.GEOM.hex_radius, separation_mm * 1e-3),
        separation_mm * 1e-3,
    ):
        rel_pos = np.maximum(rel_pos, np.linalg.norm(dofs[:, i, :3] - dofs[:, j, :3], axis=1))
        rel_rot = np.maximum(rel_rot, np.linalg.norm(dofs[:, i, 3:] - dofs[:, j, 3:], axis=1))
    return {
        "t": t,
        "mean_position_m": np.asarray(result.pos_err),
        "mean_rotation_rad": np.asarray(result.rot_err),
        "adjacent_position_m": rel_pos,
        "adjacent_rotation_rad": rel_rot,
    }


def run_transient(args, label, model, temperature, separation, count, boundary):
    old = (
        sim.GEOM.gap, sim.GEOM.mechanisms_per_edge,
        sim.GEOM.row_separation_m, sim.GEOM.mechanism_edge_half_span_fraction,
        sim.DYN.run_time_s,
    )
    sim.GEOM.gap = separation * 1e-3
    sim.GEOM.mechanisms_per_edge = count
    sim.GEOM.row_separation_m = 0.045
    sim.GEOM.mechanism_edge_half_span_fraction = HALF_SPAN_MM / (sim.GEOM.hex_radius * 1e3)
    sim.DYN.run_time_s = args.run_time
    start = time.perf_counter()
    try:
        with transient_boundary(
            boundary, separation, args.support_stiffness,
            args.kinematic_support_zeta,
        ) as state:
            result = sim.run_case(
                temperature_k=temperature,
                temperature_model=model,
                timestep=args.timestep,
                show_viewer=False,
            )
    finally:
        (
            sim.GEOM.gap, sim.GEOM.mechanisms_per_edge,
            sim.GEOM.row_separation_m, sim.GEOM.mechanism_edge_half_span_fraction,
            sim.DYN.run_time_s,
        ) = old
    histories = response_histories(result, separation)
    if not all(np.all(np.isfinite(values)) for values in histories.values()):
        raise RuntimeError(f"Nonfinite history for {label}, {boundary}.")
    history_name = f"history_{label}_{boundary}.npz"
    np.savez_compressed(args.output_dir / history_name, **histories)
    peak = lambda key, scale=1.0: float(np.max(histories[key]) * scale)
    return {
        "history_file": history_name,
        "peak_mean_position_mm": peak("mean_position_m", 1e3),
        "peak_mean_rotation_mrad": peak("mean_rotation_rad", 1e3),
        "peak_adjacent_position_mm": peak("adjacent_position_m", 1e3),
        "peak_adjacent_rotation_mrad": peak("adjacent_rotation_rad", 1e3),
        "max_mechanism_force_N": float(result.max_force),
        "mechanism_force_clip_count": int(result.clip_count),
        "max_support_displacement_m": float(state["max_displacement_m"]),
        "max_support_force_N": float(state["max_force_N"]),
        "wall_runtime_s": time.perf_counter() - start,
    }


def write_results(output: Path, rows: list[dict]) -> None:
    (output / "support_comparison_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "support_comparison_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_temperature_models(output: Path) -> None:
    T = np.linspace(50.0, 85.0, 351)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for name, label, color in (("chiang", "Chiang PCHIP", "C0"), ("jiang", "Jiang PCHIP", "C1")):
        ratio = np.array([temperature_ratio(t, name) for t in T])
        axes[0].plot(T, ratio, label=label, color=color, linewidth=2.2)
        for d, ls in ((7.5, "-"), (10.0, "--"), (25.0, ":")):
            values = np.array([mechanism_K6(d*1e-3, t, name)[1, 1] for t in T])
            axes[1].plot(T, values, color=color, linestyle=ls, linewidth=2,
                         label=f"{label}, {d:g} mm")
    chiang_mask = (CHIANG_T_K >= 50.0) & (CHIANG_T_K <= 85.0)
    axes[0].scatter(
        CHIANG_T_K[chiang_mask],
        CHIANG_FORCE_N[chiang_mask] / CHIANG_FORCE_N[np.flatnonzero(CHIANG_T_K == 77.0)[0]],
        color="C0", marker="o", facecolors="none", zorder=4,
        label="Chiang digitized points",
    )
    jiang_mask = (JIANG_T_K >= 50.0) & (JIANG_T_K <= 85.0)
    axes[0].scatter(
        JIANG_T_K[jiang_mask],
        JIANG_FORCE_N[jiang_mask] / float(jiang_pchip_force_n(77.0)),
        color="C1", marker="s", facecolors="none", zorder=4,
        label="Jiang tabulated points",
    )
    axes[0].axvline(77, color="0.4", linestyle=":", label="77 K reference")
    axes[0].set(xlabel="Temperature (K)", ylabel="Normalized multiplier",
                title="Independent temperature sensitivities")
    axes[1].set(xlabel="Temperature (K)", ylabel="Perpendicular stiffness (N/m)",
                title="Resulting fixed field-cooled stiffness")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Chiang and Jiang Temperature Models — Separately Normalized at 77 K")
    fig.savefig(output / "figure_temperature_sensitivities_separate.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, label in (("chiang", "Chiang PCHIP"), ("jiang", "Jiang PCHIP")):
        for temperature in (50.0, 77.0, 85.0):
            for separation in (7.5, 10.0, 25.0):
                K = np.diag(mechanism_K6(separation*1e-3, temperature, name))
                C = np.diag(mechanism_C6(
                    separation*1e-3, temperature, name, zeta=MECHANISM_ZETA
                ))
                rows.append({
                    "model": name, "model_label": label,
                    "temperature_K": temperature, "field_cooling_separation_mm": separation,
                    "temperature_multiplier": temperature_ratio(temperature, name),
                    "K_parallel_N_per_m": K[0], "K_perpendicular_N_per_m": K[1],
                    "C_parallel_Ns_per_m": C[0], "C_perpendicular_Ns_per_m": C[1],
                })
    with (output / "temperature_sensitivity_K_C_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_geometry(output: Path) -> None:
    pos = positions(10.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, title in zip(axes, ("1% center restraint", "3-2-1 kinematic support")):
        for idx, (x, y) in enumerate(pos):
            patch = RegularPolygon((x, y), 6, radius=sim.GEOM.hex_radius,
                                   orientation=0, facecolor="#dbe9f6", edgecolor="0.35")
            ax.add_patch(patch); ax.text(x, y, str(idx), ha="center", va="center", fontsize=12)
        ax.scatter(pos[0,0], pos[0,1], marker="*", s=180, color="#d62728", zorder=5)
        ax.annotate("Impulse: segment 0", xy=pos[0], xytext=(pos[0,0]+0.25, pos[0,1]+0.42),
                    arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
        ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
    center = int(np.argmin(np.linalg.norm(pos, axis=1)))
    axes[0].scatter(pos[center,0], pos[center,1], marker="s", s=130, color="#2ca02c", zorder=6)
    axes[0].annotate("6-DOF restraint\nsegment 3", xy=pos[center], xytext=(0.2, -0.55),
                     arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c")
    colors = ("#9467bd", "#ff7f0e", "#17becf")
    for item, color in zip(supports(10.0), colors):
        point = pos[item.segment] + item.local_point[:2]
        axes[1].scatter(*point, s=130, color=color, zorder=6)
        axes[1].annotate(f"{item.name}\nsegment {item.segment}", xy=point,
                         xytext=point + np.array([0.12, 0.18]), fontsize=9,
                         arrowprops=dict(arrowstyle="->", color=color), color=color)
    fig.suptitle("Boundary-Condition Comparison (same segment numbering and impulse)")
    fig.savefig(output / "figure_support_geometry_center1pct_vs_321.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def case_title(row: dict) -> str:
    model = "Jiang PCHIP" if row["model"] == "jiang" else "Chiang PCHIP"
    return f"{model}\n{row['temperature_K']:g} K, {row['separation_mm']:g} mm, N=4"


def make_plots(output: Path, rows: list[dict]) -> None:
    plot_temperature_models(output); plot_geometry(output)
    by_case = {label: [r for r in rows if r["case"] == label] for label, *_ in CASES}
    x = np.arange(len(CASES)); width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    center = [by_case[label][0]["center_frequency_hz"] for label, *_ in CASES]
    exact = [by_case[label][0]["kinematic_exact_frequency_hz"] for label, *_ in CASES]
    ax.bar(x-width/2, center, width, label="1% center restraint")
    ax.bar(x+width/2, exact, width, label="Exact 3-2-1 constraint (no support damping)")
    ax.set_xticks(x, [case_title(by_case[label][0]) for label, *_ in CASES])
    ax.set_ylabel("Lowest supported-mode frequency (Hz)"); ax.grid(axis="y", alpha=.25); ax.legend()
    ax.set_title("Modal Sensitivity to Support Topology")
    fig.savefig(output / "figure_321_vs_center1pct_frequency.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fields = (
        ("peak_mean_position_mm", "Peak mean position (mm)"),
        ("peak_mean_rotation_mrad", "Peak mean rotation (mrad)"),
        ("peak_adjacent_position_mm", "Peak adjacent position (mm)"),
        ("peak_adjacent_rotation_mrad", "Peak adjacent rotation (mrad)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (field, ylabel) in zip(axes.flat, fields):
        c = [next(r[field] for r in by_case[label] if r["boundary"] == "center_1pct") for label, *_ in CASES]
        k = [next(r[field] for r in by_case[label] if r["boundary"] == "kinematic_321") for label, *_ in CASES]
        ax.bar(x-width/2, c, width, label="1% center restraint")
        ax.bar(x+width/2, k, width, label="3-2-1 penalty transient")
        ax.set_xticks(x, [label.replace("_", "\n") for label, *_ in CASES])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
    axes[0,0].legend(); fig.suptitle("Peak Impulse Response: 1% Center Restraint versus 3-2-1 Support")
    fig.savefig(output / "figure_321_vs_center1pct_peak_response.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    reference = by_case["reference"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for row in reference:
        h = np.load(output / row["history_file"])
        label = "1% center restraint" if row["boundary"] == "center_1pct" else "3-2-1 penalty transient"
        for ax, key, scale, ylabel in (
            (axes[0,0], "mean_position_m", 1e3, "Mean position (mm)"),
            (axes[0,1], "mean_rotation_rad", 1e3, "Mean rotation (mrad)"),
            (axes[1,0], "adjacent_position_m", 1e3, "Adjacent position (mm)"),
            (axes[1,1], "adjacent_rotation_rad", 1e3, "Adjacent rotation (mrad)"),
        ):
            ax.plot(h["t"], scale*h[key], label=label, linewidth=1.5); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    for ax in axes[1]: ax.set_xlabel("Time (s)")
    axes[0,0].legend(); fig.suptitle("Reference-Case Histories — 77 K, 10 mm, N=4")
    fig.savefig(output / "figure_321_vs_center1pct_reference_histories.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    # These two figures and the K/C table require no transient integration and
    # are written immediately, even if a later MuJoCo case is interrupted.
    plot_temperature_models(args.output_dir)
    plot_geometry(args.output_dir)
    result_path = args.output_dir / "support_comparison_results.json"
    rows = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
    done = {(r["case"], r["boundary"]) for r in rows}
    processed = 0
    if not args.plots_only:
        for case_index, (label, model, temperature, separation, count) in enumerate(CASES, 1):
            modal = modal_results(
                model, temperature, separation, count,
                args.support_stiffness, args.kinematic_support_zeta,
            )
            for boundary in ("center_1pct", "kinematic_321"):
                key = (label, boundary)
                if key in done and not args.force:
                    print(f"SKIP {label} | {boundary}", flush=True); continue
                if args.max_cases is not None and processed >= args.max_cases:
                    write_results(args.output_dir, rows); return
                print(f"[{case_index}/4] RUN {label} | {boundary}", flush=True)
                transient = run_transient(
                    args, label, model, temperature, separation, count, boundary
                )
                row = {
                    "case": label, "model": model,
                    "temperature_model_label": "Jiang PCHIP" if model == "jiang" else "Chiang PCHIP",
                    "temperature_K": temperature, "separation_mm": separation,
                    "mechanism_count_per_interface": count, "boundary": boundary,
                    "field_cooling_matrices_fixed_during_transient": True,
                    "mechanism_damping_ratio": MECHANISM_ZETA,
                    "center_restraint_damping_ratio": CENTER_RESTRAINT_ZETA,
                    "kinematic_penalty_support_damping_ratio": args.kinematic_support_zeta,
                    "center_translation_stiffness_N_per_m": 500.0,
                    "center_rotation_stiffness_Nm_per_rad": 500.0,
                    "kinematic_penalty_stiffness_N_per_m": args.support_stiffness,
                    "center_frequency_hz": modal["center"]["frequency_hz"],
                    "center_modal_damping_ratio": modal["center"]["modal_damping_ratio"],
                    "kinematic_exact_frequency_hz": modal["kinematic_exact"]["frequency_hz"],
                    "kinematic_exact_modal_damping_ratio": modal["kinematic_exact"]["modal_damping_ratio"],
                    "kinematic_penalty_frequency_hz": modal["kinematic_penalty"]["frequency_hz"],
                    "kinematic_penalty_frequency_error_percent": 100*(modal["kinematic_penalty"]["frequency_hz"] / modal["kinematic_exact"]["frequency_hz"] - 1),
                    **transient,
                }
                rows = [r for r in rows if (r["case"], r["boundary"]) != key] + [row]
                write_results(args.output_dir, rows); done.add(key); processed += 1
    if len(rows) == 2*len(CASES):
        make_plots(args.output_dir, rows)
        print(f"Complete. Outputs: {args.output_dir}")
    else:
        print("Partial run saved; rerun the same command to resume.")


if __name__ == "__main__":
    main()
