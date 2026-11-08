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
from thesis_config import MAIN_SWEEP_DIR
from vertical_placement_study import (
    adjacent_pairs,
    build_linear_system,
    settling_time,
)
from mechanism_parameters import (
    CONFIG,
    mechanism_K6,
    temperature_ratio,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MAIN_SWEEP_DIR
DEFAULT_COUNTS = [2, 4, 6, 8, 10]
DEFAULT_TEMPERATURES = [50.0, 77.0, 85.0]
DEFAULT_SEPARATIONS = [7.5, 10.0, 25.0]
DEFAULT_MODELS = ["chiang", "jiang"]
BASELINE_Z_MM = 22.5
BASELINE_ROW_SEPARATION_MM = 45.0
BASELINE_HALF_SPAN_MM = 225.0


def case_key(model: str, temperature: float, separation: float, count: int) -> str:
    return f"{model.lower()}|{temperature:g}|{separation:g}|{count:d}"


def along_edge_positions(count: int, half_span_mm: float) -> list[float]:
    if count < 2 or count % 2:
        raise ValueError("Mechanism count must be even and at least 2.")
    per_row = count // 2
    if per_row == 1:
        return [0.0]
    return np.linspace(-half_span_mm, half_span_mm, per_row).tolist()


def base_row(model: str, temperature: float, separation: float, count: int,
             half_span_mm: float) -> dict:
    return {
        "model": model,
        "temperature_K": float(temperature),
        "separation_mm": float(separation),
        "mechanism_count_per_interface": int(count),
        "mechanisms_per_row": int(count // 2),
        "along_edge_positions_each_row_mm": along_edge_positions(count, half_span_mm),
        "along_edge_half_span_mm": float(half_span_mm),
        "row_half_height_z_mm": BASELINE_Z_MM,
        "row_separation_mm": BASELINE_ROW_SEPARATION_MM,
        "surface_inset_mm": 2.5,
        "modal_complete": False,
        "dynamic_complete": False,
    }


def modal_metrics(model: str, temperature: float, separation: float, count: int,
                  half_span_mm: float) -> dict:
    K, C, M = build_linear_system(
        temperature, model, count, separation, BASELINE_Z_MM, half_span_mm
    )
    eigenvalues, eigenvectors = eigh(K, M)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)
    phi = eigenvectors[:, 0]
    omega = math.sqrt(max(float(eigenvalues[0]), 0.0))
    modal_mass = float(phi @ M @ phi)
    modal_damping = float(phi @ C @ phi)
    zeta = (
        float(modal_damping / (2.0 * omega * modal_mass))
        if omega > 0.0 and modal_mass > 0.0 else None
    )
    linear_settling = (
        float(4.0 / (zeta * omega))
        if zeta is not None and zeta > 0.0 and omega > 0.0 else None
    )
    return {
        "minimum_generalized_eigenvalue": float(eigenvalues.min()),
        "first_mode_hz": float(frequencies[0]),
        "first_mode_period_s": (
            None if frequencies[0] <= 0.0 else float(1.0 / frequencies[0])
        ),
        "first_mode_damping_ratio": zeta,
        "linear_first_mode_2pct_settling_s": linear_settling,
        "first_six_frequencies_hz": frequencies[:6].tolist(),
        "modal_complete": True,
    }


def selected_runtime(args: argparse.Namespace, f1: float) -> float:
    if args.runtime_mode == "fixed" or f1 <= 0.0:
        return float(args.run_time)
    target = sim.DYN.impulse_time + 1.25 / f1 + 1.0
    return float(min(args.max_run_time, max(args.run_time, target)))


def peak_value_and_time(t: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    index = int(np.argmax(values))
    return float(values[index]), float(t[index])


def run_dynamic(args: argparse.Namespace, row: dict,
                output_dir: Path) -> dict:
    model = str(row["model"])
    temperature = float(row["temperature_K"])
    separation = float(row["separation_mm"])
    count = int(row["mechanism_count_per_interface"])
    f1 = float(row["first_mode_hz"])
    runtime = selected_runtime(args, f1)

    old = (
        sim.GEOM.gap,
        sim.GEOM.mechanisms_per_edge,
        sim.GEOM.row_separation_m,
        sim.GEOM.mechanism_edge_half_span_fraction,
        sim.DYN.run_time_s,
    )
    sim.GEOM.gap = separation * 1e-3
    sim.GEOM.mechanisms_per_edge = count
    sim.GEOM.row_separation_m = BASELINE_ROW_SEPARATION_MM * 1e-3
    sim.GEOM.mechanism_edge_half_span_fraction = (
        args.half_span_mm / (sim.GEOM.hex_radius * 1e3)
    )
    sim.DYN.run_time_s = runtime
    start = time.perf_counter()
    try:
        result = sim.run_case(
            temperature_k=temperature,
            temperature_model=model,
            show_viewer=False,
            timestep=args.timestep,
        )
        clamp_count = int(sim.DYN.separation_clamp_count)
        max_gap_excursion = float(sim.DYN.max_normal_gap_excursion_m)
        max_local_translation = float(sim.DYN.max_local_translation_m)
        min_gap = float(sim.DYN.min_instantaneous_gap_m)
        max_gap = float(sim.DYN.max_instantaneous_gap_m)
    finally:
        (
            sim.GEOM.gap,
            sim.GEOM.mechanisms_per_edge,
            sim.GEOM.row_separation_m,
            sim.GEOM.mechanism_edge_half_span_fraction,
            sim.DYN.run_time_s,
        ) = old
    wall = time.perf_counter() - start

    t = np.asarray(result.t)
    dofs = np.asarray(result.segment_dofs, dtype=float)
    seg_pos = np.linalg.norm(dofs[:, :, :3], axis=2)
    seg_rot = np.linalg.norm(dofs[:, :, 3:], axis=2)
    worst_pos = np.max(seg_pos, axis=1)
    worst_rot = np.max(seg_rot, axis=1)
    positions = sim.hex_positions(1, sim.GEOM.hex_radius, separation * 1e-3)
    pairs = adjacent_pairs(positions, separation * 1e-3)
    rel_pos = np.zeros(len(t))
    rel_rot = np.zeros(len(t))
    for i, j in pairs:
        rel_pos = np.maximum(
            rel_pos, np.linalg.norm(dofs[:, i, :3] - dofs[:, j, :3], axis=1)
        )
        rel_rot = np.maximum(
            rel_rot, np.linalg.norm(dofs[:, i, 3:] - dofs[:, j, 3:], axis=1)
        )

    pmp, tmp = peak_value_and_time(t, result.pos_err)
    pmr, tmr = peak_value_and_time(t, result.rot_err)
    pwp, twp = peak_value_and_time(t, worst_pos)
    pwr, twr = peak_value_and_time(t, worst_rot)
    prp, trp = peak_value_and_time(t, rel_pos)
    prr, trr = peak_value_and_time(t, rel_rot)
    peak_limit = 0.90 * runtime

    save_history = (
        model == "chiang"
        and math.isclose(temperature, 50.0)
        and math.isclose(separation, 7.5)
    )
    history_file = None
    if save_history:
        history_file = (
            f"history_structural_{model}_T{temperature:g}K_d{separation:g}mm_"
            f"N{count}_z22p5mm.npz"
        )
        np.savez_compressed(
            output_dir / history_file,
            t=t,
            mean_position_error_m=np.asarray(result.pos_err),
            mean_rotation_error_rad=np.asarray(result.rot_err),
            worst_segment_position_error_m=worst_pos,
            worst_segment_rotation_error_rad=worst_rot,
            max_adjacent_relative_position_error_m=rel_pos,
            max_adjacent_relative_rotation_error_rad=rel_rot,
        )

    return {
        "requested_minimum_runtime_s": float(args.run_time),
        "used_runtime_s": runtime,
        "runtime_mode": args.runtime_mode,
        "peak_mean_position_error_m": pmp,
        "peak_mean_position_time_s": tmp,
        "peak_mean_rotation_error_rad": pmr,
        "peak_mean_rotation_time_s": tmr,
        "peak_worst_segment_position_error_m": pwp,
        "peak_worst_segment_position_time_s": twp,
        "peak_worst_segment_rotation_error_rad": pwr,
        "peak_worst_segment_rotation_time_s": twr,
        "peak_adjacent_relative_position_error_m": prp,
        "peak_adjacent_relative_position_time_s": trp,
        "peak_adjacent_relative_rotation_error_rad": prr,
        "peak_adjacent_relative_rotation_time_s": trr,
        "mean_position_2pct_settling_after_impulse_s": settling_time(
            t, np.asarray(result.pos_err), sim.DYN.impulse_time
        ),
        "mean_rotation_2pct_settling_after_impulse_s": settling_time(
            t, np.asarray(result.rot_err), sim.DYN.impulse_time
        ),
        "all_peaks_captured_before_final_10pct": bool(
            max(tmp, tmr, twp, twr, trp, trr) < peak_limit
        ),
        "max_mechanism_force_N": float(result.max_force),
        "max_mechanism_torque_Nm": float(result.max_torque),
        "clip_count": int(result.clip_count),
        "instantaneous_gap_outside_diagnostic_domain_count": clamp_count,
        "max_normal_gap_excursion_m": max_gap_excursion,
        "max_normal_gap_excursion_fraction_of_field_cooling_gap": (
            max_gap_excursion / (separation * 1e-3)
        ),
        "max_local_mechanism_translation_m": max_local_translation,
        "min_instantaneous_gap_m": min_gap,
        "max_instantaneous_gap_m": max_gap,
        "small_motion_within_2mm_empirical_translation_range": (
            max_local_translation <= 0.002
        ),
        "normal_excursion_within_declared_10pct_engineering_check": (
            max_gap_excursion <= 0.10 * separation * 1e-3
        ),
        "field_cooling_constitutive_matrices_fixed_during_transient": True,
        "wall_runtime_s": float(wall),
        "history_file": history_file,
        "dynamic_complete": True,
    }


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        case_key(str(r["model"]), float(r["temperature_K"]),
                 float(r["separation_mm"]), int(r["mechanism_count_per_interface"])): r
        for r in data
    }


def write_rows(output_dir: Path, rows: dict[str, dict]) -> None:
    ordered = sorted(rows.values(), key=lambda r: (
        str(r["model"]), int(r["mechanism_count_per_interface"]),
        float(r["temperature_K"]), float(r["separation_mm"]),
    ))
    (output_dir / "main_sweep_z22p5mm.json").write_text(
        json.dumps(ordered, indent=2), encoding="utf-8"
    )
    if not ordered:
        return
    fields: list[str] = []
    for row in ordered:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_dir / "main_sweep_z22p5mm.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def plot_geometry(output_dir: Path, counts: list[int], half_span_mm: float) -> None:
    fig, axes = plt.subplots(len(counts), 1, figsize=(12, 2.05 * len(counts)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    edge_half = 0.5 * sim.GEOM.hex_radius * 1e3
    for ax, count in zip(axes, counts):
        s = np.asarray(along_edge_positions(count, half_span_mm))
        for z, color in ((-BASELINE_Z_MM, "C0"), (BASELINE_Z_MM, "C1")):
            ax.plot([-edge_half, edge_half], [z, z], color="0.65")
            ax.scatter(s, np.full_like(s, z), color=color, s=75, zorder=3)
        ax.axvline(0.0, color="0.5", linestyle=":")
        ax.set_ylim(-32, 32)
        ax.set_yticks([-22.5, 22.5], labels=["-22.5", "+22.5"])
        ax.set_title(f"N={count}: {count//2} mechanisms per row; s={s.tolist()} mm",
                     fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)
    axes[-1].set_xlim(-edge_half - 10, edge_half + 10)
    axes[-1].set_xlabel("Along-edge coordinate, s (mm)")
    axes[len(axes)//2].set_ylabel("Vertical row position, z (mm)")
    fig.suptitle(
        "Mechanism Placement: +/-225 mm Along Edge, z=+/-22.5 mm",
        fontsize=16,
    )
    fig.savefig(output_dir / "final_mechanism_geometry_z22p5mm.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def matrix(rows: list[dict], model: str, count: int, field: str,
           temperatures: list[float], separations: list[float],
           scale: float = 1.0) -> np.ndarray:
    out = np.full((len(temperatures), len(separations)), np.nan)
    for row in rows:
        if str(row["model"]) != model:
            continue
        if int(row["mechanism_count_per_interface"]) != count:
            continue
        try:
            i = temperatures.index(float(row["temperature_K"]))
            j = separations.index(float(row["separation_mm"]))
        except ValueError:
            continue
        value = row.get(field)
        if value is not None:
            out[i, j] = float(value) * scale
    return out


def plot_heatmap_family(output_dir: Path, rows: list[dict], models: list[str],
                        counts: list[int], temperatures: list[float],
                        separations: list[float], field: str, scale: float,
                        label: str, stem: str) -> None:
    for model in models:
        mats = [matrix(rows, model, n, field, temperatures, separations, scale)
                for n in counts]
        finite_chunks = [m[np.isfinite(m)] for m in mats if np.isfinite(m).any()]
        if not finite_chunks:
            continue
        finite = np.concatenate(finite_chunks)
        vmin, vmax = float(finite.min()), float(finite.max())
        fig, axes = plt.subplots(1, len(counts), figsize=(3.4*len(counts)+1.2, 4.3),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for ax, count, values in zip(axes, counts, mats):
            image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis",
                              vmin=vmin, vmax=vmax)
            ax.set_title(f"N={count}")
            ax.set_xticks(range(len(separations)), labels=[f"{x:g}" for x in separations])
            ax.set_yticks(range(len(temperatures)), labels=[f"{x:g}" for x in temperatures])
            ax.set_xlabel("Field-cooling separation (mm)")
        axes[0].set_ylabel("Temperature (K)")
        fig.colorbar(image, ax=axes, label=label, shrink=0.82)
        fig.suptitle(
            f"{model.title()} {label} versus Mechanism Count\n"
            "Baseline: z=+/-22.5 mm, along-edge half-span=225 mm",
            fontsize=15,
        )
        fig.savefig(output_dir / f"final_{model}_{stem}_z22p5mm.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def plot_count_trends(output_dir: Path, rows: list[dict], models: list[str],
                      counts: list[int]) -> None:
    if not any(bool(row.get("dynamic_complete", False)) for row in rows):
        return
    fields = [
        ("first_mode_hz", 1.0, "First natural frequency (Hz)"),
        ("peak_mean_position_error_m", 1e3, "Peak mean position error (mm)"),
        ("peak_mean_rotation_error_rad", 1e3, "Peak mean rotation error (mrad)"),
        ("peak_adjacent_relative_rotation_error_rad", 1e3,
         "Peak adjacent relative rotation (mrad)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (field, scale, ylabel) in zip(axes.flat, fields):
        for model in models:
            values = []
            for count in counts:
                match = next((r for r in rows
                              if str(r["model"]) == model
                              and int(r["mechanism_count_per_interface"]) == count
                              and math.isclose(float(r["temperature_K"]), 50.0)
                              and math.isclose(float(r["separation_mm"]), 7.5)
                              and r.get(field) is not None), None)
                values.append(np.nan if match is None else float(match[field]) * scale)
            ax.plot(counts, values, marker="o", label=model.title())
        ax.set_xlabel("Mechanisms per interface, N")
        ax.set_ylabel(ylabel)
        ax.set_xticks(counts)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Mechanism-Count Trends at 50 K and 7.5 mm Separation\n"
        "Final z=+/-22.5 mm baseline",
        fontsize=16,
    )
    fig.savefig(output_dir / "final_count_trends_T50K_d7p5mm_z22p5mm.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_parameter_models(output_dir: Path) -> None:
    separations = np.linspace(5.0, 30.0, 180)
    temps = [50.0, 77.0, 85.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for temperature in temps:
        kp, kn = [], []
        for d in separations:
            K = mechanism_K6(d * 1e-3, temperature, "chiang")
            kp.append(K[0, 0]); kn.append(K[1, 1])
        axes[0].plot(separations, kp, label=f"{temperature:g} K")
        axes[1].plot(separations, kn, label=f"{temperature:g} K")
    axes[0].set_title("Parallel stiffness")
    axes[1].set_title("Perpendicular stiffness")
    for ax in axes:
        ax.set_xlabel("Field-cooling separation (mm)")
        ax.set_ylabel("Stiffness per mechanism (N/m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Chiang Directional Stiffness Model", fontsize=16)
    fig.savefig(output_dir / "final_directional_stiffness_model.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    t = np.linspace(50.0, 85.0, 180)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for model, label in (("chiang", "Chiang"), ("jiang", "Jiang")):
        ax.plot(t, [temperature_ratio(x, model) for x in t], label=label)
    ax.axvline(77.0, color="0.4", linestyle=":", label="77 K reference")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalized stiffness multiplier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Temperature Scaling Models Normalized at 77 K")
    fig.savefig(output_dir / "final_temperature_scaling_models.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_saved_histories(output_dir: Path, rows: list[dict]) -> None:
    selected = [r for r in rows if r.get("history_file")]
    if not selected:
        return
    selected.sort(key=lambda r: int(r["mechanism_count_per_interface"]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             constrained_layout=True)
    for row in selected:
        path = output_dir / str(row["history_file"])
        if not path.exists():
            continue
        with np.load(path) as data:
            label = f"N={int(row['mechanism_count_per_interface'])}"
            axes[0].plot(data["t"], data["mean_position_error_m"] * 1e3,
                         label=label)
            axes[1].plot(data["t"], data["mean_rotation_error_rad"] * 1e3,
                         label=label)
    axes[0].set_ylabel("Mean position error (mm)")
    axes[1].set_ylabel("Mean rotation error (mrad)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(sim.DYN.impulse_time, color="0.4", linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Impulse Response versus Mechanism Count: Chiang, 50 K, 7.5 mm\n"
        "Final z=+/-22.5 mm baseline",
        fontsize=16,
    )
    fig.savefig(output_dir / "final_count_response_histories_z22p5mm.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def make_all_plots(output_dir: Path, rows: dict[str, dict], args: argparse.Namespace) -> None:
    data = list(rows.values())
    plot_geometry(output_dir, args.counts, args.half_span_mm)
    plot_parameter_models(output_dir)
    families = [
        ("first_mode_hz", 1.0, "First natural frequency (Hz)", "frequency"),
        ("peak_mean_position_error_m", 1e3, "Peak mean position error (mm)", "mean_position"),
        ("peak_mean_rotation_error_rad", 1e3, "Peak mean rotation error (mrad)", "mean_rotation"),
        ("peak_adjacent_relative_position_error_m", 1e3,
         "Peak adjacent relative position error (mm)", "relative_position"),
        ("peak_adjacent_relative_rotation_error_rad", 1e3,
         "Peak adjacent relative rotation error (mrad)", "relative_rotation"),
    ]
    for field, scale, label, stem in families:
        plot_heatmap_family(
            output_dir, data, args.models, args.counts, args.temperatures,
            args.separations_mm, field, scale, label, stem,
        )
    plot_count_trends(output_dir, data, args.models, args.counts)
    plot_saved_histories(output_dir, data)


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "study": "current field-cooling-fixed dual-temperature structural results suite",
        "scope": "Fundamental frequency, position error, rotation error, relative alignment, and settling",
        "wavefront_error_scope": "Excluded",
        "final_vertical_baseline_z_mm": [-BASELINE_Z_MM, BASELINE_Z_MM],
        "final_row_separation_mm": BASELINE_ROW_SEPARATION_MM,
        "surface_inset_mm": 2.5,
        "along_edge_half_span_mm": args.half_span_mm,
        "models": args.models,
        "mechanism_counts": args.counts,
        "temperatures_K": args.temperatures,
        "separations_mm": args.separations_mm,
        "runtime_mode": args.runtime_mode,
        "minimum_runtime_s": args.run_time,
        "maximum_runtime_s": args.max_run_time,
        "timestep_s": args.timestep,
        "impulse_force_N": list(sim.DYN.impulse_force),
        "impulse_duration_s": sim.DYN.impulse_duration,
        "impulse_point_offset_m": list(sim.DYN.impulse_point_offset),
        "center_restraint_translation_stiffness_N_per_m": sim.DYN.anchor_k_trans,
        "center_restraint_rotation_stiffness_Nm_per_rad": sim.DYN.anchor_k_rot,
        "center_restraint_translation_damping_ratio": sim.DYN.anchor_zeta_trans,
        "center_restraint_rotation_damping_ratio": sim.DYN.anchor_zeta_rot,
        "temperature_model_interpretation": {
            "chiang": "independent shape-preserving PCHIP interpolation of digitized Chiang data, normalized at 77 K",
            "jiang": "independent shape-preserving PCHIP interpolation of Jiang tabulated levitation-force data, normalized at 77 K",
        },
        "note": (
            "Results from the former z=+/-25 mm baseline are not mixed with this dataset. "
            "All modal and dynamic results in this output use z=+/-22.5 mm."
        ),
    }
    (output_dir / "main_sweep_z22p5mm_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="current field-cooling-fixed dual-temperature structural sweep at z=+/-22.5 mm with 1% center damping.")
    parser.add_argument("--phase", choices=("modal", "dynamic", "both", "plots"), default="modal")
    parser.add_argument("--models", nargs="+", choices=("chiang", "jiang"), default=DEFAULT_MODELS)
    parser.add_argument("--counts", nargs="+", type=int, default=DEFAULT_COUNTS)
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--separations-mm", nargs="+", type=float, default=DEFAULT_SEPARATIONS)
    parser.add_argument("--half-span-mm", type=float, default=BASELINE_HALF_SPAN_MM)
    parser.add_argument("--run-time", type=float, default=15.0)
    parser.add_argument("--max-run-time", type=float, default=60.0)
    parser.add_argument("--runtime-mode", choices=("fixed", "one-period"), default="one-period")
    parser.add_argument("--timestep", type=float, default=2e-4)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models = list(dict.fromkeys(args.models))
    args.counts = sorted(set(args.counts))
    args.temperatures = sorted(set(args.temperatures))
    args.separations_mm = sorted(set(args.separations_mm))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "main_sweep_z22p5mm.json"
    rows = load_rows(results_path)
    save_metadata(args.output_dir, args)

    plan = [
        (model, temperature, separation, count)
        for count in args.counts
        for model in args.models
        for temperature in args.temperatures
        for separation in args.separations_mm
    ]
    print("=" * 100)
    print("current FC-fixed DUAL-TEMPERATURE SWEEP: z=+/-22.5 mm, 1% CENTER DAMPING")
    print("=" * 100)
    print(f"models={args.models} | counts={args.counts}")
    print(f"temperatures={args.temperatures} | separations={args.separations_mm} mm")
    print("row positions=[-22.5,+22.5] mm | row separation=45 mm | surface inset=2.5 mm")
    print(f"phase={args.phase} | cases={len(plan)} | optical wavefront metrics excluded")
    if args.phase in {"dynamic", "both"}:
        print("Dynamic runs are resumable; each incomplete case is a full MuJoCo integration.")

    processed = 0
    for index, (model, temperature, separation, count) in enumerate(plan, start=1):
        key = case_key(model, temperature, separation, count)
        row = rows.get(key) or base_row(
            model, temperature, separation, count, args.half_span_mm
        )
        if args.phase == "plots":
            rows[key] = row
            continue
        need_modal = args.force or not bool(row.get("modal_complete", False))
        need_dynamic = args.force or not bool(row.get("dynamic_complete", False))
        action = (
            (args.phase == "modal" and need_modal)
            or (args.phase == "dynamic" and need_dynamic)
            or (args.phase == "both" and (need_modal or need_dynamic))
        )
        if not action:
            print(f"[{index}/{len(plan)}] SKIP {model} N={count} T={temperature:g} K d={separation:g} mm")
            rows[key] = row
            continue
        if args.max_cases is not None and processed >= args.max_cases:
            break
        if args.phase in {"modal", "both"} and need_modal:
            print(f"[{index}/{len(plan)}] MODAL {model} N={count} T={temperature:g} K d={separation:g} mm")
            row.update(modal_metrics(model, temperature, separation, count, args.half_span_mm))
            rows[key] = row
            write_rows(args.output_dir, rows)
        if args.phase in {"dynamic", "both"} and need_dynamic:
            if not bool(row.get("modal_complete", False)):
                row.update(modal_metrics(model, temperature, separation, count, args.half_span_mm))
            print(f"[{index}/{len(plan)}] DYNAMIC {model} N={count} T={temperature:g} K d={separation:g} mm")
            row.update(run_dynamic(args, row, args.output_dir))
            rows[key] = row
            write_rows(args.output_dir, rows)
            diagnostic_valid = (
                row["small_motion_within_2mm_empirical_translation_range"]
                and row["normal_excursion_within_declared_10pct_engineering_check"]
                and row["all_peaks_captured_before_final_10pct"]
            )
            status = "VALID" if diagnostic_valid else "REVIEW"
            print(
                f"    {status} | f1={row['first_mode_hz']:.5f} Hz | "
                f"pos={row['peak_mean_position_error_m']*1e3:.4f} mm | "
                f"rot={row['peak_mean_rotation_error_rad']*1e3:.4f} mrad | "
                f"sim={row['used_runtime_s']:.1f}s | wall={row['wall_runtime_s']:.1f}s"
            )
        processed += 1

    write_rows(args.output_dir, rows)
    make_all_plots(args.output_dir, rows, args)
    summary = {
        "planned_cases": len(plan),
        "rows_in_output": len(rows),
        "modal_complete": sum(bool(r.get("modal_complete", False)) for r in rows.values()),
        "dynamic_complete": sum(bool(r.get("dynamic_complete", False)) for r in rows.values()),
        "review_cases": sum(
            bool(r.get("dynamic_complete", False))
            and not (
                bool(r.get("small_motion_within_2mm_empirical_translation_range", False))
                and bool(r.get("normal_excursion_within_declared_10pct_engineering_check", False))
                and bool(r.get("all_peaks_captured_before_final_10pct", False))
            )
            for r in rows.values()
        ),
        "output_directory": str(args.output_dir.resolve()),
    }
    (args.output_dir / "main_sweep_z22p5mm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Complete | modal={summary['modal_complete']} | dynamic={summary['dynamic_complete']} | review={summary['review_cases']}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
