from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

import segment_dynamics as sim
from thesis_config import VERTICAL_DIR
from mechanism_parameters import CONFIG, mechanism_C6, mechanism_K6, temperature_ratio


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = VERTICAL_DIR
DEFAULT_Z_MM = [22.5, 20.0, 17.5]
DEFAULT_TEMPERATURE_K = 50.0
DEFAULT_SEPARATION_MM = 7.5
DEFAULT_MECHANISM_COUNT = 10
DEFAULT_HALF_SPAN_MM = 225.0
DEFAULT_RUNTIME_S = 15.0


def z_key(z_mm: float) -> str:
    return f"{float(z_mm):g}"


def z_tag(z_mm: float) -> str:
    return f"{float(z_mm):g}".replace(".", "p")


def validate_inputs(z_values_mm: list[float], mechanism_count: int,
                    half_span_mm: float) -> None:
    half_thickness_mm = 0.5 * CONFIG.segment.thickness_m * 1e3
    if not z_values_mm:
        raise ValueError("At least one positive z value is required.")
    if any(z <= 0.0 or z >= half_thickness_mm for z in z_values_mm):
        raise ValueError(
            f"Every z value must lie strictly inside (0, {half_thickness_mm:g}) mm."
        )
    if mechanism_count < 2 or mechanism_count % 2:
        raise ValueError("Mechanism count must be even and at least 2.")
    edge_half_mm = 0.5 * sim.GEOM.hex_radius * 1e3
    if half_span_mm <= 0.0 or half_span_mm > edge_half_mm:
        raise ValueError(
            f"Along-edge half-span must lie in (0, {edge_half_mm:g}] mm."
        )


def adjacent_pairs(positions: list[tuple[float, float]], gap_m: float) -> list[tuple[int, int]]:
    pos = np.asarray(positions, dtype=float)
    threshold = 1.15 * (math.sqrt(3.0) * sim.GEOM.hex_radius + gap_m)
    return [
        (i, j)
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if np.linalg.norm(pos[j] - pos[i]) < threshold
    ]


def skew(r: np.ndarray) -> np.ndarray:
    x, y, z = map(float, r)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def build_linear_system(temperature_k: float, model: str, mechanism_count: int,
                        separation_mm: float, z_mm: float,
                        half_span_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the Field-cooling-fixed linear system for symmetric rows at z=+/-z_mm."""
    gap_m = separation_mm * 1e-3
    positions = sim.hex_positions(1, sim.GEOM.hex_radius, gap_m)
    pos = np.asarray(positions, dtype=float)
    pairs = adjacent_pairs(positions, gap_m)
    nbody = len(positions)
    ndof = 6 * nbody
    K = np.zeros((ndof, ndof), dtype=float)
    C = np.zeros_like(K)
    M = np.zeros_like(K)
    M6 = np.diag(np.r_[
        [CONFIG.segment.mass_kg] * 3,
        CONFIG.segment.inertia_diag_kgm2,
    ])
    for i in range(nbody):
        sl = slice(6 * i, 6 * i + 6)
        M[sl, sl] = M6

    # Retain the complete local constitutive matrices.  The primary model has
    # zero direct rotational rows/columns, so this gives the same primary
    # assembly as the former 3x3 path.  It also correctly assembles comparison
    # matrices containing direct rotational or translation-rotation terms.
    K6 = np.asarray(
        mechanism_K6(gap_m, temperature_k, model), dtype=float
    )
    C6 = np.asarray(
        mechanism_C6(
            gap_m, temperature_k, model, zeta=CONFIG.zeta_nominal
        ),
        dtype=float,
    )
    if K6.shape != (6, 6) or C6.shape != (6, 6):
        raise ValueError("Local mechanism K and C matrices must both be 6x6.")
    edge_offset = math.sqrt(3.0) * 0.5 * sim.GEOM.hex_radius
    per_row = mechanism_count // 2
    half_span_m = half_span_mm * 1e-3
    s_values = (
        np.array([0.0])
        if per_row == 1
        else np.linspace(-half_span_m, half_span_m, per_row)
    )
    row_offsets_m = (-z_mm * 1e-3, z_mm * 1e-3)

    for i, j in pairs:
        dvec = pos[j] - pos[i]
        n2 = dvec / np.linalg.norm(dvec)
        t2 = np.array([-n2[1], n2[0]])
        T = np.column_stack((
            np.array([t2[0], t2[1], 0.0]),
            np.array([n2[0], n2[1], 0.0]),
            np.array([0.0, 0.0, -1.0]),
        ))
        for z_offset in row_offsets_m:
            for s in s_values:
                ai2 = n2 * edge_offset + t2 * s
                aj2 = -n2 * edge_offset + t2 * s
                ai = np.array([ai2[0], ai2[1], z_offset])
                aj = np.array([aj2[0], aj2[1], z_offset])
                # Local relative translation at the two attachment points.
                Bi_translation = T.T @ np.block([np.eye(3), -skew(ai)])
                Bj_translation = T.T @ np.block([-np.eye(3), skew(aj)])

                # Small relative rotation, expressed in the same local frame.
                Bi_rotation = np.block([np.zeros((3, 3)), T.T])
                Bj_rotation = np.block([np.zeros((3, 3)), -T.T])

                B = np.zeros((6, ndof), dtype=float)
                B[:, 6 * i:6 * i + 6] = np.vstack((
                    Bi_translation, Bi_rotation,
                ))
                B[:, 6 * j:6 * j + 6] = np.vstack((
                    Bj_translation, Bj_rotation,
                ))
                K += B.T @ K6 @ B
                C += B.T @ C6 @ B

    center_idx = int(np.argmin([math.hypot(x, y) for x, y in positions]))
    sl = slice(6 * center_idx, 6 * center_idx + 6)
    k_t = sim.DYN.anchor_k_trans
    k_r = sim.DYN.anchor_k_rot
    c_t = 2.0 * sim.DYN.anchor_zeta_trans * math.sqrt(
        k_t * sim.DYN.anchor_mass
    )
    c_r = 2.0 * sim.DYN.anchor_zeta_rot * math.sqrt(
        k_r * CONFIG.segment.inertia_x_kgm2
    )
    K[sl, sl] += np.diag([k_t] * 3 + [k_r] * 3)
    C[sl, sl] += np.diag([c_t] * 3 + [c_r] * 3)
    return 0.5 * (K + K.T), 0.5 * (C + C.T), M


def modal_result(temperature_k: float, model: str, mechanism_count: int,
                 separation_mm: float, z_mm: float,
                 half_span_mm: float) -> dict:
    K, _C, M = build_linear_system(
        temperature_k, model, mechanism_count, separation_mm, z_mm,
        half_span_mm,
    )
    eigenvalues, _eigenvectors = eigh(K, M)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)
    return {
        "minimum_generalized_eigenvalue": float(eigenvalues.min()),
        "first_mode_hz": float(frequencies[0]),
        "first_mode_period_s": (
            None if frequencies[0] <= 0.0 else float(1.0 / frequencies[0])
        ),
        "first_six_frequencies_hz": frequencies[:6].tolist(),
    }


def base_row(args: argparse.Namespace, z_mm: float) -> dict:
    half_thickness_mm = 0.5 * CONFIG.segment.thickness_m * 1e3
    largest_z = max(args.z_mm)
    return {
        "model": args.model,
        "temperature_K": float(args.temperature),
        "separation_mm": float(args.separation_mm),
        "mechanism_count_per_interface": int(args.mechanism_count),
        "mechanisms_per_row": int(args.mechanism_count // 2),
        "along_edge_half_span_mm": float(args.half_span_mm),
        "row_half_height_z_mm": float(z_mm),
        "row_positions_mm": [-float(z_mm), float(z_mm)],
        "row_separation_mm": 2.0 * float(z_mm),
        "surface_inset_mm": half_thickness_mm - float(z_mm),
        "idealized_rotational_stiffness_ratio_to_largest_z": (
            float(z_mm) / float(largest_z)
        ) ** 2,
        "segment_thickness_mm": CONFIG.segment.thickness_m * 1e3,
        "stiffness_model": "Field-cooling-fixed primary pure-translational K11/K33 fits",
        "damping_model": f"Field-cooling-fixed mechanism-level zeta={CONFIG.zeta_nominal:g}",
        "modal_complete": False,
        "dynamic_complete": False,
    }


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return {z_key(float(row["row_half_height_z_mm"])): row for row in data}


def write_rows(output_dir: Path, rows: dict[str, dict]) -> None:
    ordered = sorted(
        rows.values(), key=lambda row: float(row["row_half_height_z_mm"]),
        reverse=True,
    )
    json_path = output_dir / "vertical_placement_results.json"
    csv_path = output_dir / "vertical_placement_results.csv"
    json_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    if not ordered:
        return
    fields: list[str] = []
    for row in ordered:
        for name in row:
            if name not in fields:
                fields.append(name)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def settling_time(t: np.ndarray, values: np.ndarray, impulse_time: float,
                  fraction: float = 0.02) -> float | None:
    post = np.flatnonzero(t >= impulse_time)
    if post.size == 0:
        return None
    peak_local = int(np.argmax(values[post]))
    peak_index = int(post[peak_local])
    peak = float(values[peak_index])
    if peak <= 0.0:
        return 0.0
    threshold = fraction * peak
    suffix_max = np.maximum.accumulate(values[::-1])[::-1]
    candidates = np.flatnonzero(
        (np.arange(len(t)) >= peak_index) & (suffix_max <= threshold)
    )
    if candidates.size == 0:
        return None
    return float(t[int(candidates[0])] - impulse_time)


def dynamic_result(args: argparse.Namespace, z_mm: float,
                   output_dir: Path) -> dict:
    old_gap = sim.GEOM.gap
    old_count = sim.GEOM.mechanisms_per_edge
    old_row_separation = sim.GEOM.row_separation_m
    old_fraction = sim.GEOM.mechanism_edge_half_span_fraction
    old_runtime = sim.DYN.run_time_s
    sim.GEOM.gap = args.separation_mm * 1e-3
    sim.GEOM.mechanisms_per_edge = int(args.mechanism_count)
    sim.GEOM.row_separation_m = 2.0 * z_mm * 1e-3
    sim.GEOM.mechanism_edge_half_span_fraction = (
        args.half_span_mm / (sim.GEOM.hex_radius * 1e3)
    )
    sim.DYN.run_time_s = float(args.run_time)
    start = time.perf_counter()
    try:
        result = sim.run_case(
            temperature_k=float(args.temperature),
            temperature_model=args.model,
            show_viewer=False,
            timestep=float(args.timestep),
        )
    finally:
        sim.GEOM.gap = old_gap
        sim.GEOM.mechanisms_per_edge = old_count
        sim.GEOM.row_separation_m = old_row_separation
        sim.GEOM.mechanism_edge_half_span_fraction = old_fraction
        sim.DYN.run_time_s = old_runtime

    elapsed = time.perf_counter() - start
    dofs = np.asarray(result.segment_dofs, dtype=float)
    segment_position = np.linalg.norm(dofs[:, :, :3], axis=2)
    segment_rotation = np.linalg.norm(dofs[:, :, 3:], axis=2)
    worst_position = np.max(segment_position, axis=1)
    worst_rotation = np.max(segment_rotation, axis=1)
    positions = sim.hex_positions(
        1, sim.GEOM.hex_radius, args.separation_mm * 1e-3
    )
    pairs = adjacent_pairs(positions, args.separation_mm * 1e-3)
    relative_position = np.zeros(len(result.t), dtype=float)
    relative_rotation = np.zeros(len(result.t), dtype=float)
    for i, j in pairs:
        relative_position = np.maximum(
            relative_position,
            np.linalg.norm(dofs[:, i, :3] - dofs[:, j, :3], axis=1),
        )
        relative_rotation = np.maximum(
            relative_rotation,
            np.linalg.norm(dofs[:, i, 3:] - dofs[:, j, 3:], axis=1),
        )

    history_path = output_dir / f"history_vertical_z_{z_tag(z_mm)}mm.npz"
    np.savez_compressed(
        history_path,
        t=np.asarray(result.t),
        mean_position_error_m=np.asarray(result.pos_err),
        mean_rotation_error_rad=np.asarray(result.rot_err),
        worst_segment_position_error_m=worst_position,
        worst_segment_rotation_error_rad=worst_rotation,
        max_adjacent_relative_position_error_m=relative_position,
        max_adjacent_relative_rotation_error_rad=relative_rotation,
        mean_absolute_dof_error=np.asarray(result.dof_err),
    )
    impulse_time = float(sim.DYN.impulse_time)
    return {
        "peak_mean_position_error_m": float(np.max(result.pos_err)),
        "peak_mean_rotation_error_rad": float(np.max(result.rot_err)),
        "peak_worst_segment_position_error_m": float(np.max(worst_position)),
        "peak_worst_segment_rotation_error_rad": float(np.max(worst_rotation)),
        "peak_adjacent_relative_position_error_m": float(np.max(relative_position)),
        "peak_adjacent_relative_rotation_error_rad": float(np.max(relative_rotation)),
        "mean_position_2pct_settling_after_impulse_s": settling_time(
            result.t, result.pos_err, impulse_time
        ),
        "mean_rotation_2pct_settling_after_impulse_s": settling_time(
            result.t, result.rot_err, impulse_time
        ),
        "used_runtime_s": float(args.run_time),
        "wall_runtime_s": float(elapsed),
        "max_mechanism_force_N": float(result.max_force),
        "max_mechanism_torque_Nm": float(result.max_torque),
        "clip_count": int(result.clip_count),
        "instantaneous_gap_outside_diagnostic_domain_count": int(
            sim.DYN.separation_clamp_count
        ),
        "max_normal_gap_excursion_m": float(sim.DYN.max_normal_gap_excursion_m),
        "max_normal_gap_excursion_fraction_of_field_cooling_gap": float(
            sim.DYN.max_normal_gap_excursion_m / max(sim.GEOM.gap, 1e-12)
        ),
        "max_local_mechanism_translation_m": float(
            sim.DYN.max_local_translation_m
        ),
        "min_instantaneous_gap_m": float(sim.DYN.min_instantaneous_gap_m),
        "max_instantaneous_gap_m": float(sim.DYN.max_instantaneous_gap_m),
        "field_cooling_constitutive_matrices_fixed_during_transient": True,
        "history_file": history_path.name,
        "dynamic_complete": True,
    }


def make_geometry_plot(output_dir: Path, z_values_mm: list[float]) -> None:
    half_t = 0.5 * CONFIG.segment.thickness_m * 1e3
    fig, axes = plt.subplots(
        1, len(z_values_mm), figsize=(4.4 * len(z_values_mm), 5.2),
        sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, z_mm in zip(axes, z_values_mm):
        ax.fill_between([-1.0, 1.0], -half_t, half_t, color="0.92")
        ax.axhline(half_t, color="0.25", linewidth=1.5)
        ax.axhline(-half_t, color="0.25", linewidth=1.5)
        ax.scatter([0.0, 0.0], [-z_mm, z_mm], s=130, color=["C0", "C1"], zorder=3)
        ax.plot([0.0, 0.0], [-z_mm, z_mm], color="0.45", linestyle=":")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([-25, -20, -15, 0, 15, 20, 25])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(
            f"z = +/-{z_mm:g} mm\nrow separation = {2*z_mm:g} mm\n"
            f"surface inset = {half_t-z_mm:g} mm"
        )
    axes[0].set_ylabel("Through-thickness coordinate, z (mm)")
    fig.suptitle("Symmetric Vertical Mechanism-Row Placements", fontsize=16)
    fig.savefig(output_dir / "vertical_height_geometry.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def make_plots(output_dir: Path, rows: dict[str, dict]) -> None:
    completed_modal = [
        row for row in rows.values() if bool(row.get("modal_complete", False))
    ]
    completed_modal.sort(key=lambda row: float(row["row_half_height_z_mm"]))
    if completed_modal:
        z = np.array([float(row["row_half_height_z_mm"]) for row in completed_modal])
        f1 = np.array([float(row["first_mode_hz"]) for row in completed_modal])
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        ax.plot(z, f1, marker="o", linewidth=2.2)
        ax.set_xlabel("Mechanism-row half-height, |z| (mm)")
        ax.set_ylabel("First natural frequency (Hz)")
        ax.set_xticks(z)
        ax.grid(True, alpha=0.3)
        ax.set_title("Fundamental Frequency versus Vertical Row Placement")
        fig.savefig(output_dir / "vertical_height_frequency.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    completed_dynamic = [
        row for row in rows.values() if bool(row.get("dynamic_complete", False))
    ]
    completed_dynamic.sort(key=lambda row: float(row["row_half_height_z_mm"]))
    if not completed_dynamic:
        return
    z = np.array([float(row["row_half_height_z_mm"]) for row in completed_dynamic])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(z, [float(r["peak_mean_position_error_m"]) * 1e3 for r in completed_dynamic],
                 marker="o", label="Assembly mean")
    axes[0].plot(z, [float(r["peak_worst_segment_position_error_m"]) * 1e3 for r in completed_dynamic],
                 marker="s", label="Worst segment")
    axes[0].set_ylabel("Peak position error (mm)")
    axes[1].plot(z, [float(r["peak_mean_rotation_error_rad"]) * 1e3 for r in completed_dynamic],
                 marker="o", label="Assembly mean")
    axes[1].plot(z, [float(r["peak_worst_segment_rotation_error_rad"]) * 1e3 for r in completed_dynamic],
                 marker="s", label="Worst segment")
    axes[1].set_ylabel("Peak rotation error (mrad)")
    for ax in axes:
        ax.set_xlabel("Mechanism-row half-height, |z| (mm)")
        ax.set_xticks(z)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Peak Structural Response versus Vertical Row Placement", fontsize=16)
    fig.savefig(output_dir / "vertical_height_peak_errors.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(z, [float(r["peak_adjacent_relative_position_error_m"]) * 1e3 for r in completed_dynamic], marker="o")
    axes[0].set_ylabel("Peak adjacent relative position error (mm)")
    axes[1].plot(z, [float(r["peak_adjacent_relative_rotation_error_rad"]) * 1e3 for r in completed_dynamic], marker="o")
    axes[1].set_ylabel("Peak adjacent relative rotation error (mrad)")
    for ax in axes:
        ax.set_xlabel("Mechanism-row half-height, |z| (mm)")
        ax.set_xticks(z)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Worst Adjacent-Segment Relative Response", fontsize=16)
    fig.savefig(output_dir / "vertical_height_relative_errors.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             constrained_layout=True)
    for row in completed_dynamic:
        history = output_dir / str(row["history_file"])
        if not history.exists():
            continue
        with np.load(history) as data:
            label = f"z=+/-{float(row['row_half_height_z_mm']):g} mm"
            axes[0].plot(data["t"], data["mean_position_error_m"] * 1e3,
                         label=label)
            axes[1].plot(data["t"], data["mean_rotation_error_rad"] * 1e3,
                         label=label)
    axes[0].set_ylabel("Mean position error (mm)")
    axes[1].set_ylabel("Mean rotation error (mrad)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(sim.DYN.impulse_time, color="0.35", linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Structural Response Histories versus Vertical Row Placement", fontsize=16)
    fig.savefig(output_dir / "vertical_height_time_histories.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "study": "Field-cooling-fixed symmetric vertical mechanism-row placement sensitivity",
        "scope": "Structural metrics only; no wavefront-error calculation",
        "model": args.model,
        "temperature_K": args.temperature,
        "separation_mm": args.separation_mm,
        "mechanism_count_per_interface": args.mechanism_count,
        "mechanisms_per_row": args.mechanism_count // 2,
        "along_edge_half_span_mm": args.half_span_mm,
        "z_half_heights_mm": args.z_mm,
        "row_separations_mm": [2.0 * z for z in args.z_mm],
        "surface_insets_mm": [25.0 - z for z in args.z_mm],
        "segment_thickness_mm": CONFIG.segment.thickness_m * 1e3,
        "runtime_s": args.run_time,
        "timestep_s": args.timestep,
        "impulse_force_N": list(sim.DYN.impulse_force),
        "impulse_duration_s": sim.DYN.impulse_duration,
        "impulse_point_offset_m": list(sim.DYN.impulse_point_offset),
        "anchor_k_translation_N_per_m": sim.DYN.anchor_k_trans,
        "anchor_k_rotation_Nm_per_rad": sim.DYN.anchor_k_rot,
        "settling_definition": (
            "First time after the post-impulse peak for which the complete "
            "remaining raw history stays at or below 2% of that peak. Null "
            "means the criterion was not demonstrated within the run."
        ),
        "placement_note": (
            "Rows remain symmetric about the segment mid-plane. Changing z "
            "changes only the through-thickness moment arm; the along-edge "
            "positions, mechanism matrices, count, impulse, and anchor are fixed."
        ),
    }
    (output_dir / "vertical_placement_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Field-cooling-fixed vertical mechanism-row height sensitivity study."
    )
    parser.add_argument("--phase", choices=("modal", "dynamic", "both", "plots"),
                        default="modal")
    parser.add_argument("--z-mm", nargs="+", type=float, default=DEFAULT_Z_MM)
    parser.add_argument("--model", choices=("chiang", "jiang"), default="chiang")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--separation-mm", type=float, default=DEFAULT_SEPARATION_MM)
    parser.add_argument("--mechanism-count", type=int, default=DEFAULT_MECHANISM_COUNT)
    parser.add_argument("--half-span-mm", type=float, default=DEFAULT_HALF_SPAN_MM)
    parser.add_argument("--run-time", type=float, default=DEFAULT_RUNTIME_S)
    parser.add_argument("--timestep", type=float, default=2.0e-4)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.z_mm = sorted(set(float(z) for z in args.z_mm), reverse=True)
    validate_inputs(args.z_mm, args.mechanism_count, args.half_span_mm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "vertical_placement_results.json"
    rows = load_rows(json_path)
    save_metadata(args.output_dir, args)
    make_geometry_plot(args.output_dir, args.z_mm)

    print("=" * 96)
    print("Field-cooling-fixed VERTICAL MECHANISM-ROW HEIGHT SENSITIVITY")
    print("=" * 96)
    print(
        f"model={args.model} | T={args.temperature:g} K | "
        f"gap={args.separation_mm:g} mm | N={args.mechanism_count} | "
        f"along-edge half-span=+/-{args.half_span_mm:g} mm"
    )
    print(f"z half-heights={args.z_mm} mm")
    print(f"row separations={[2.0*z for z in args.z_mm]} mm")
    print("outputs: fundamental frequency, position error, rotation error, and settling")
    print("wavefront error: excluded from this study")
    print(f"phase={args.phase}")

    processed = 0
    for index, z_mm in enumerate(args.z_mm, start=1):
        key = z_key(z_mm)
        row = rows.get(key) or base_row(args, z_mm)
        needs_modal = args.force or not bool(row.get("modal_complete", False))
        needs_dynamic = args.force or not bool(row.get("dynamic_complete", False))
        action_needed = (
            (args.phase == "modal" and needs_modal)
            or (args.phase == "dynamic" and needs_dynamic)
            or (args.phase == "both" and (needs_modal or needs_dynamic))
        )
        if args.phase == "plots":
            rows[key] = row
            continue
        if not action_needed:
            print(f"[{index}/{len(args.z_mm)}] SKIP z=+/-{z_mm:g} mm")
            rows[key] = row
            continue
        if args.max_cases is not None and processed >= args.max_cases:
            break
        if args.phase in {"modal", "both"} and needs_modal:
            print(f"[{index}/{len(args.z_mm)}] MODAL z=+/-{z_mm:g} mm")
            row.update(modal_result(
                args.temperature, args.model, args.mechanism_count,
                args.separation_mm, z_mm, args.half_span_mm,
            ))
            row["modal_complete"] = True
            print(f"    f1={row['first_mode_hz']:.6f} Hz")
            rows[key] = row
            write_rows(args.output_dir, rows)
        if args.phase in {"dynamic", "both"} and needs_dynamic:
            if not bool(row.get("modal_complete", False)):
                row.update(modal_result(
                    args.temperature, args.model, args.mechanism_count,
                    args.separation_mm, z_mm, args.half_span_mm,
                ))
                row["modal_complete"] = True
            print(f"[{index}/{len(args.z_mm)}] DYNAMIC z=+/-{z_mm:g} mm")
            row.update(dynamic_result(args, z_mm, args.output_dir))
            print(
                f"    complete | f1={row['first_mode_hz']:.6f} Hz | "
                f"peak mean position={row['peak_mean_position_error_m']*1e3:.6f} mm | "
                f"peak mean rotation={row['peak_mean_rotation_error_rad']*1e3:.6f} mrad | "
                f"wall={row['wall_runtime_s']:.1f} s"
            )
            rows[key] = row
            write_rows(args.output_dir, rows)
        processed += 1

    write_rows(args.output_dir, rows)
    make_plots(args.output_dir, rows)
    summary = {
        "requested_z_half_heights_mm": args.z_mm,
        "rows_in_output": len(rows),
        "modal_complete": sum(bool(r.get("modal_complete", False)) for r in rows.values()),
        "dynamic_complete": sum(bool(r.get("dynamic_complete", False)) for r in rows.values()),
        "output_directory": str(args.output_dir.resolve()),
    }
    (args.output_dir / "vertical_placement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Complete | modal={summary['modal_complete']} | "
        f"dynamic={summary['dynamic_complete']}"
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
