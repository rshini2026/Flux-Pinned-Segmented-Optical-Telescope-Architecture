"""Adaptive settling comparison for the primary and alternate FPI matrices.

The study evaluates:

1. Chiang, 50 K, 7.5 mm field-cooling separation, N=10.
2. Jiang, 50 K, 7.5 mm field-cooling separation, N=10.
3. The common Chiang/Jiang-normalized 77 K, 25 mm, N=4 condition.

For each condition, the full alternate 6-DOF representation is run first.  A
case is declared settled when mean position, mean rotation, maximum adjacent
position, and maximum adjacent rotation have all remained below 2% of their
own post-impulse peaks for ``--hold-time`` seconds.  The primary and alternate
translations-only representations are then run to the same final time so all
three histories have a common comparison window.

The published alternate matrix is used directly at 77 K and 25 mm.  Its use
at 50 K and 7.5 mm is explicitly exploratory.  Translational terms are scaled
direction-by-direction using the selected primary temperature/separation
model.  Because the source provides no independent rotational scaling law,
the direct rotational stiffnesses use the geometric mean of the two
translational stiffness scale factors; direct rotational damping is scaled by
the square root of that factor.  The chosen factors are written to every
result file.

The script is resumable at the completed-case level. Re-running the same
command skips completed histories and regenerates the summary and plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

import segment_dynamics as sim
import vertical_placement_study as vertical
from thesis_config import (
    BASELINE_ALONG_EDGE_HALF_SPAN_MM,
    BASELINE_ROW_HALF_HEIGHT_MM,
    CENTER_RESTRAINT_DAMPING_RATIO,
    DEFAULT_TIMESTEP_S,
    IMPULSE_TIME_S,
    MECHANISM_DAMPING_RATIO,
)
from mechanism_parameters import (
    CONFIG,
    K_parallel,
    K_perp,
    mechanism_C6,
    mechanism_K6,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "Alternate_Matrix_Settling_Comparison_Output"

ALT_K_TRANSLATION = np.diag([5.40, 8.77, 5.40, 0.0, 0.0, 0.0])
ALT_C_TRANSLATION = np.diag([0.0113, 0.0149, 0.0113, 0.0, 0.0, 0.0])
ALT_K_FULL = np.diag([5.40, 8.77, 5.40, 292.0, 0.0, 358.0])
ALT_C_FULL = np.diag([0.0113, 0.0149, 0.0113, 0.4306, 0.0, 0.4042])
REFERENCE_TEMPERATURE_K = 77.0
REFERENCE_SEPARATION_MM = 25.0

REPRESENTATIONS = (
    ("primary", "Primary model"),
    ("alternate_translation", "Alternate translations only"),
    ("alternate_full", "Alternate full 6-DOF"),
)
METRICS = (
    ("mean_position_m", "Mean position error", "mm", 1e3),
    ("mean_rotation_rad", "Mean rotation error", "mrad", 1e3),
    ("adjacent_position_m", "Maximum adjacent position error", "mm", 1e3),
    ("adjacent_rotation_rad", "Maximum adjacent rotation error", "mrad", 1e3),
)


def slug_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def group_key(model: str, temperature_k: float, separation_mm: float,
              count: int) -> str:
    model_tag = "normalized" if temperature_k == 77.0 else model
    return (
        f"{model_tag}_T{slug_number(temperature_k)}K_"
        f"d{slug_number(separation_mm)}mm_N{count}"
    )


def history_name(group: str, representation: str) -> str:
    return f"history_{group}_{representation}.npz"


def adjacent_pairs(separation_mm: float) -> list[tuple[int, int]]:
    positions = sim.hex_positions(1, sim.GEOM.hex_radius, separation_mm * 1e-3)
    return vertical.adjacent_pairs(positions, separation_mm * 1e-3)


def adjacent_metrics(dofs: np.ndarray,
                     pairs: list[tuple[int, int]]) -> tuple[float, float]:
    rel_position = 0.0
    rel_rotation = 0.0
    for i, j in pairs:
        rel_position = max(
            rel_position,
            float(np.linalg.norm(dofs[i, :3] - dofs[j, :3])),
        )
        rel_rotation = max(
            rel_rotation,
            float(np.linalg.norm(dofs[i, 3:] - dofs[j, 3:])),
        )
    return rel_position, rel_rotation


@dataclass
class SettlementMonitor:
    pairs: list[tuple[int, int]]
    fraction: float
    hold_time_s: float
    impulse_time_s: float
    impulse_duration_s: float
    minimum_observation_s: float = 5.0

    def __post_init__(self) -> None:
        self.peaks = np.zeros(4, dtype=float)
        self.last_exceedance_s: float | None = None
        self.settling_start_s: float | None = None
        self.stop_time_s: float | None = None

    def __call__(self, t: float, dofs: np.ndarray,
                 mean_position: float, mean_rotation: float) -> bool:
        if t < self.impulse_time_s:
            return False
        adjacent_position, adjacent_rotation = adjacent_metrics(dofs, self.pairs)
        values = np.array([
            mean_position, mean_rotation,
            adjacent_position, adjacent_rotation,
        ], dtype=float)
        self.peaks = np.maximum(self.peaks, values)
        thresholds = self.fraction * self.peaks
        # A newly observed peak or any threshold exceedance restarts the dwell.
        if np.any(values > thresholds) or np.any(self.peaks <= 0.0):
            self.last_exceedance_s = float(t)
            return False
        earliest = (
            self.impulse_time_s + self.impulse_duration_s
            + self.minimum_observation_s
        )
        if t < earliest or self.last_exceedance_s is None:
            return False
        if t - self.last_exceedance_s >= self.hold_time_s:
            self.settling_start_s = float(self.last_exceedance_s)
            self.stop_time_s = float(t)
            return True
        return False


def scaling_factors(model: str, temperature_k: float,
                    separation_mm: float) -> dict[str, float]:
    target_gap = separation_mm * 1e-3
    reference_gap = REFERENCE_SEPARATION_MM * 1e-3
    target_k = mechanism_K6(target_gap, temperature_k, model)
    reference_k = mechanism_K6(
        reference_gap, REFERENCE_TEMPERATURE_K, model
    )
    target_c = mechanism_C6(
        target_gap, temperature_k, model, zeta=MECHANISM_DAMPING_RATIO
    )
    reference_c = mechanism_C6(
        reference_gap, REFERENCE_TEMPERATURE_K, model,
        zeta=MECHANISM_DAMPING_RATIO,
    )
    k_parallel_scale = float(target_k[0, 0] / reference_k[0, 0])
    k_perpendicular_scale = float(target_k[1, 1] / reference_k[1, 1])
    c_parallel_scale = float(target_c[0, 0] / reference_c[0, 0])
    c_perpendicular_scale = float(target_c[1, 1] / reference_c[1, 1])
    k_rotational_scale = math.sqrt(
        k_parallel_scale * k_perpendicular_scale
    )
    c_rotational_scale = math.sqrt(k_rotational_scale)
    return {
        "k_parallel_scale": k_parallel_scale,
        "k_perpendicular_scale": k_perpendicular_scale,
        "c_parallel_scale": c_parallel_scale,
        "c_perpendicular_scale": c_perpendicular_scale,
        "k_rotational_scale": k_rotational_scale,
        "c_rotational_scale": c_rotational_scale,
    }


def local_matrices(model: str, temperature_k: float, separation_mm: float,
                   representation: str) -> tuple[np.ndarray, np.ndarray, dict]:
    gap_m = separation_mm * 1e-3
    factors = scaling_factors(model, temperature_k, separation_mm)
    if representation == "primary":
        K = mechanism_K6(gap_m, temperature_k, model)
        C = mechanism_C6(
            gap_m, temperature_k, model, zeta=MECHANISM_DAMPING_RATIO
        )
        policy = "Final primary fixed field-cooling matrices"
    else:
        exact_reference = (
            math.isclose(temperature_k, REFERENCE_TEMPERATURE_K)
            and math.isclose(separation_mm, REFERENCE_SEPARATION_MM)
        )
        K = np.array(
            ALT_K_FULL if representation == "alternate_full"
            else ALT_K_TRANSLATION,
            dtype=float,
        )
        C = np.array(
            ALT_C_FULL if representation == "alternate_full"
            else ALT_C_TRANSLATION,
            dtype=float,
        )
        if not exact_reference:
            K[0, 0] *= factors["k_parallel_scale"]
            K[1, 1] *= factors["k_perpendicular_scale"]
            K[2, 2] *= factors["k_parallel_scale"]
            C[0, 0] *= factors["c_parallel_scale"]
            C[1, 1] *= factors["c_perpendicular_scale"]
            C[2, 2] *= factors["c_parallel_scale"]
            if representation == "alternate_full":
                K[3:, 3:] *= factors["k_rotational_scale"]
                C[3:, 3:] *= factors["c_rotational_scale"]
            policy = (
                "Exploratory alternate-matrix scaling from the published "
                "77 K, 25 mm reference"
            )
        else:
            policy = "Published alternate matrix used at its reference condition"
    details = {**factors, "matrix_policy": policy,
               "local_K6": K.tolist(), "local_C6": C.tolist()}
    return K, C, details


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def linear_matrices(model_name: str, temperature_k: float,
                    separation_mm: float, count: int,
                    local_K: np.ndarray,
                    local_C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble 42-DOF matrices including direct local rotational terms."""
    positions = sim.hex_positions(1, sim.GEOM.hex_radius, separation_mm * 1e-3)
    pairs = vertical.adjacent_pairs(positions, separation_mm * 1e-3)
    ndof = 6 * len(positions)
    M = np.zeros((ndof, ndof))
    K = np.zeros((ndof, ndof))
    C = np.zeros((ndof, ndof))
    inertia = CONFIG.segment.inertia_diag_kgm2
    for body in range(len(positions)):
        block = slice(6 * body, 6 * body + 6)
        M[block, block] = np.diag(
            [CONFIG.segment.mass_kg] * 3 + inertia.tolist()
        )
    edge_offset = math.sqrt(3.0) * 0.5 * sim.GEOM.hex_radius
    per_row = count // 2
    s_values = (
        np.array([0.0]) if per_row == 1
        else np.linspace(
            -BASELINE_ALONG_EDGE_HALF_SPAN_MM * 1e-3,
            BASELINE_ALONG_EDGE_HALF_SPAN_MM * 1e-3,
            per_row,
        )
    )
    z_values = (
        -BASELINE_ROW_HALF_HEIGHT_MM * 1e-3,
        BASELINE_ROW_HALF_HEIGHT_MM * 1e-3,
    )
    for i, j in pairs:
        pi = np.asarray(positions[i], dtype=float)
        pj = np.asarray(positions[j], dtype=float)
        n2 = (pj - pi) / np.linalg.norm(pj - pi)
        t2 = np.array([-n2[1], n2[0]])
        T = np.column_stack((
            np.array([t2[0], t2[1], 0.0]),
            np.array([n2[0], n2[1], 0.0]),
            np.array([0.0, 0.0, -1.0]),
        ))
        for z_offset in z_values:
            for s_value in s_values:
                ai2 = n2 * edge_offset + t2 * s_value
                aj2 = -n2 * edge_offset + t2 * s_value
                ai = np.array([ai2[0], ai2[1], z_offset])
                aj = np.array([aj2[0], aj2[1], z_offset])
                B = np.zeros((6, ndof))
                B[:3, 6*i:6*i+6] = T.T @ np.block([
                    np.eye(3), -skew(ai)
                ])
                B[:3, 6*j:6*j+6] = T.T @ np.block([
                    -np.eye(3), skew(aj)
                ])
                B[3:, 6*i+3:6*i+6] = T.T
                B[3:, 6*j+3:6*j+6] = -T.T
                K += B.T @ local_K @ B
                C += B.T @ local_C @ B
    center = int(np.argmin([math.hypot(x, y) for x, y in positions]))
    block = slice(6 * center, 6 * center + 6)
    c_t = 2.0 * CENTER_RESTRAINT_DAMPING_RATIO * math.sqrt(
        sim.DYN.anchor_k_trans * sim.DYN.anchor_mass
    )
    c_r = 2.0 * CENTER_RESTRAINT_DAMPING_RATIO * math.sqrt(
        sim.DYN.anchor_k_rot * CONFIG.segment.inertia_x_kgm2
    )
    K[block, block] += np.diag(
        [sim.DYN.anchor_k_trans] * 3 + [sim.DYN.anchor_k_rot] * 3
    )
    C[block, block] += np.diag([c_t] * 3 + [c_r] * 3)
    return 0.5*(K+K.T), 0.5*(C+C.T), M


def modal_summary(model_name: str, temperature_k: float,
                  separation_mm: float, count: int,
                  local_K: np.ndarray, local_C: np.ndarray) -> dict:
    K, C, M = linear_matrices(
        model_name, temperature_k, separation_mm, count, local_K, local_C
    )
    values, vectors = eigh(K, M)
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    frequencies = np.sqrt(values) / (2.0 * np.pi)
    damping = []
    for index in range(min(6, len(values))):
        omega = math.sqrt(values[index])
        phi = vectors[:, index]
        modal_mass = float(phi @ M @ phi)
        modal_c = float(phi @ C @ phi)
        damping.append(
            None if omega <= 0.0 else modal_c/(2.0*omega*modal_mass)
        )
    return {
        "first_six_frequencies_hz": frequencies[:6].tolist(),
        "first_six_damping_ratios": damping,
    }


def settling_metrics(t: np.ndarray, values: np.ndarray,
                     fraction: float) -> dict:
    mask = t >= IMPULSE_TIME_S
    tp = np.asarray(t[mask], dtype=float)
    yp = np.asarray(values[mask], dtype=float)
    peak = float(np.max(yp))
    threshold = fraction * peak
    remaining = np.maximum.accumulate(yp[::-1])[::-1]
    valid = np.flatnonzero(remaining <= threshold)
    absolute = float(tp[valid[0]]) if len(valid) else None
    return {
        "peak": peak,
        "threshold": threshold,
        "settled_within_record": bool(len(valid)),
        "settling_absolute_time_s": absolute,
        "settling_after_impulse_s": (
            None if absolute is None else absolute - IMPULSE_TIME_S
        ),
        "final_fraction_of_peak": (
            float(yp[-1]/peak) if peak > 0.0 else 0.0
        ),
    }


def run_transient(*, output: Path, group: str, model_name: str,
                  temperature_k: float, separation_mm: float, count: int,
                  representation: str, local_K: np.ndarray,
                  local_C: np.ndarray, matrix_details: dict,
                  run_time_s: float, timestep_s: float, log_stride: int,
                  monitor: SettlementMonitor | None, force: bool) -> dict:
    path = output / history_name(group, representation)
    if path.exists() and not force:
        with np.load(path, allow_pickle=True) as saved:
            return json.loads(saved["metadata_json"].item())
    old = {
        "gap": sim.GEOM.gap,
        "count": sim.GEOM.mechanisms_per_edge,
        "row": sim.GEOM.row_separation_m,
        "half": sim.GEOM.mechanism_edge_half_span_fraction,
        "runtime": sim.DYN.run_time_s,
        "zeta": sim.DYN.zeta_nominal,
        "anchor_t": sim.DYN.anchor_zeta_trans,
        "anchor_r": sim.DYN.anchor_zeta_rot,
    }
    sim.GEOM.gap = separation_mm * 1e-3
    sim.GEOM.mechanisms_per_edge = count
    sim.GEOM.row_separation_m = 2.0 * BASELINE_ROW_HALF_HEIGHT_MM * 1e-3
    sim.GEOM.mechanism_edge_half_span_fraction = (
        BASELINE_ALONG_EDGE_HALF_SPAN_MM / (sim.GEOM.hex_radius * 1e3)
    )
    sim.DYN.run_time_s = run_time_s
    sim.DYN.zeta_nominal = MECHANISM_DAMPING_RATIO
    sim.DYN.anchor_zeta_trans = CENTER_RESTRAINT_DAMPING_RATIO
    sim.DYN.anchor_zeta_rot = CENTER_RESTRAINT_DAMPING_RATIO
    original_k = sim.mechanism_K6
    original_c = sim.mechanism_C6
    sim.mechanism_K6 = lambda *_args, **_kwargs: local_K
    sim.mechanism_C6 = lambda *_args, **_kwargs: local_C
    started = time.perf_counter()
    try:
        result = sim.run_case(
            temperature_k=temperature_k,
            temperature_model=model_name,
            show_viewer=False,
            timestep=timestep_s,
            stop_callback=monitor,
            log_stride=log_stride,
        )
    finally:
        sim.mechanism_K6 = original_k
        sim.mechanism_C6 = original_c
        sim.GEOM.gap = old["gap"]
        sim.GEOM.mechanisms_per_edge = old["count"]
        sim.GEOM.row_separation_m = old["row"]
        sim.GEOM.mechanism_edge_half_span_fraction = old["half"]
        sim.DYN.run_time_s = old["runtime"]
        sim.DYN.zeta_nominal = old["zeta"]
        sim.DYN.anchor_zeta_trans = old["anchor_t"]
        sim.DYN.anchor_zeta_rot = old["anchor_r"]
    t = np.asarray(result.t, dtype=float)
    dofs = np.asarray(result.segment_dofs, dtype=float)
    pairs = adjacent_pairs(separation_mm)
    adjacent_position = np.zeros(len(t))
    adjacent_rotation = np.zeros(len(t))
    for index, current in enumerate(dofs):
        adjacent_position[index], adjacent_rotation[index] = adjacent_metrics(
            current, pairs
        )
    histories = {
        "t": t,
        "mean_position_m": np.asarray(result.pos_err, dtype=float),
        "mean_rotation_rad": np.asarray(result.rot_err, dtype=float),
        "adjacent_position_m": adjacent_position,
        "adjacent_rotation_rad": adjacent_rotation,
    }
    metadata = {
        "group": group,
        "model": model_name,
        "temperature_K": temperature_k,
        "field_cooling_separation_mm": separation_mm,
        "mechanism_count_per_interface": count,
        "representation": representation,
        "history_file": path.name,
        "requested_max_runtime_s": run_time_s,
        "actual_final_time_s": float(t[-1]),
        "timestep_s": timestep_s,
        "log_stride": log_stride,
        "mechanism_damping_ratio": MECHANISM_DAMPING_RATIO,
        "center_restraint_damping_ratio": CENTER_RESTRAINT_DAMPING_RATIO,
        "field_cooling_matrices_fixed_during_transient": True,
        "wall_runtime_s": float(time.perf_counter()-started),
        "max_mechanism_force_N": float(result.max_force),
        "max_mechanism_torque_Nm": float(result.max_torque),
        "force_clip_count": int(result.clip_count),
        **matrix_details,
    }
    individual_settling = {}
    for key, values in histories.items():
        if key == "t":
            continue
        individual_settling[key] = settling_metrics(t, values, 0.02)
    metadata["settling_metrics"] = individual_settling
    metadata["all_metrics_settled_within_record"] = all(
        item["settled_within_record"]
        for item in individual_settling.values()
    )
    if monitor is not None:
        metadata["online_settling_confirmation"] = {
            "fraction": monitor.fraction,
            "hold_time_s": monitor.hold_time_s,
            "settling_start_absolute_s": monitor.settling_start_s,
            "stop_time_absolute_s": monitor.stop_time_s,
            "confirmed": monitor.stop_time_s is not None,
        }
    metadata["modal"] = modal_summary(
        model_name, temperature_k, separation_mm, count, local_K, local_C
    )
    np.savez_compressed(
        path,
        **histories,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return metadata


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items()
                if not isinstance(v, (dict, list))}
        for metric, values in row.get("settling_metrics", {}).items():
            for key, value in values.items():
                flat[f"{metric}_{key}"] = value
        flat_rows.append(flat)
    if flat_rows:
        fields = list(dict.fromkeys(
            key for row in flat_rows for key in row
        ))
        with path.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)


def plot_group(output: Path, group: str, rows: list[dict]) -> None:
    selected = [row for row in rows if row["group"] == group]
    by_rep = {row["representation"]: row for row in selected}
    if set(by_rep) != {item[0] for item in REPRESENTATIONS}:
        return
    example = selected[0]
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), sharex=True, constrained_layout=True
    )
    for representation, label in REPRESENTATIONS:
        row = by_rep[representation]
        with np.load(output / row["history_file"]) as history:
            for ax, (key, title, units, scale) in zip(axes.flat, METRICS):
                ax.plot(
                    history["t"] - IMPULSE_TIME_S,
                    scale * history[key],
                    linewidth=1.1,
                    label=label,
                )
                ax.set_title(title)
                ax.set_ylabel(units)
                ax.grid(alpha=0.25)
                ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1)
    for ax in axes[1]:
        ax.set_xlabel("Time after impulse (s)")
    axes[0, 0].legend()
    model_label = (
        "Chiang/Jiang (identical at 77 K)"
        if example["temperature_K"] == 77.0
        else example["model"].capitalize()
    )
    fig.suptitle(
        f"Settling Comparison — {model_label}\n"
        f"{example['temperature_K']:g} K, "
        f"{example['field_cooling_separation_mm']:g} mm field-cooling "
        f"separation, N = {example['mechanism_count_per_interface']}"
    )
    fig.savefig(
        output / f"figure_{group}_settling_histories.png",
        dpi=250, bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for representation, label in REPRESENTATIONS:
        row = by_rep[representation]
        with np.load(output / row["history_file"]) as history:
            t = np.asarray(history["t"])
            mask = t >= IMPULSE_TIME_S
            # The reverse cumulative maximum has an artificial downturn near
            # the record endpoint because too little future history remains.
            # Right-censor the final first-mode period of each representation.
            f1 = float(row["modal"]["first_six_frequencies_hz"][0])
            cutoff = float(t[-1]) - 1.0 / f1
            for ax, (key, title, _units, _scale) in zip(axes.flat, METRICS):
                values = np.asarray(history[key])[mask]
                envelope = np.maximum.accumulate(values[::-1])[::-1]
                peak = float(np.max(values))
                normalized = envelope / peak if peak > 0 else envelope
                visible = np.flatnonzero(t[mask] <= cutoff)
                # Preserve the envelope while keeping figure rendering bounded
                # for long, densely logged transient histories.
                plotted = visible[::10]
                ax.semilogy(
                    t[mask][plotted]-IMPULSE_TIME_S,
                    np.maximum(normalized[plotted], 1e-6),
                    linewidth=1.25,
                    label=label,
                )
                ax.set_title(title)
                ax.grid(alpha=0.25, which="both")
                ax.axhline(0.02, color="0.3", linestyle=":")
    for ax in axes[1]:
        ax.set_xlabel("Time after impulse (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Remaining envelope / peak")
    axes[0, 0].legend()
    fig.suptitle(f"Normalized Settling Envelopes — {model_label}")
    fig.savefig(
        output / f"figure_{group}_normalized_envelopes.png",
        dpi=250, bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP_S)
    parser.add_argument("--max-runtime", type=float, default=600.0)
    parser.add_argument("--hold-time", type=float, default=20.0)
    parser.add_argument("--settling-fraction", type=float, default=0.02)
    parser.add_argument(
        "--log-stride", type=int, default=5,
        help="Store one sample per this many integration steps.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--configuration",
        choices=("all", "77K_25mm_N4", "chiang_50K_7p5mm_N10",
                 "jiang_50K_7p5mm_N10"),
        default="all",
        help="Run all planned groups or only one requested configuration.",
    )
    return parser.parse_args()


def planned_groups() -> list[dict]:
    return [
        {"model": "chiang", "temperature_K": 50.0,
         "separation_mm": 7.5, "count": 10},
        {"model": "jiang", "temperature_K": 50.0,
         "separation_mm": 7.5, "count": 10},
        # At 77 K the two temperature models are normalized to the same value,
        # so one run represents both without duplicating computation.
        {"model": "chiang", "temperature_K": 77.0,
         "separation_mm": 25.0, "count": 4},
    ]


def selected_groups(configuration: str) -> list[dict]:
    groups = planned_groups()
    if configuration == "all":
        return groups
    selectors = {
        "77K_25mm_N4": lambda item: (
            item["temperature_K"] == 77.0
            and item["separation_mm"] == 25.0
            and item["count"] == 4
        ),
        "chiang_50K_7p5mm_N10": lambda item: (
            item["model"] == "chiang"
            and item["temperature_K"] == 50.0
            and item["separation_mm"] == 7.5
            and item["count"] == 10
        ),
        "jiang_50K_7p5mm_N10": lambda item: (
            item["model"] == "jiang"
            and item["temperature_K"] == 50.0
            and item["separation_mm"] == 7.5
            and item["count"] == 10
        ),
    }
    return [item for item in groups if selectors[configuration](item)]


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "alternate_matrix_settling_results.json"
    rows = load_rows(summary_path)
    completed = {
        (row["group"], row["representation"]): row for row in rows
    }
    groups = selected_groups(args.configuration)
    if args.dry_run:
        for spec in groups:
            group = group_key(
                spec["model"], spec["temperature_K"],
                spec["separation_mm"], spec["count"],
            )
            print(f"{group}:")
            for representation, _label in REPRESENTATIONS:
                K, C, details = local_matrices(
                    spec["model"], spec["temperature_K"],
                    spec["separation_mm"], representation,
                )
                modal = modal_summary(
                    spec["model"], spec["temperature_K"],
                    spec["separation_mm"], spec["count"], K, C,
                )
                print(
                    f"  {representation}: f1={modal['first_six_frequencies_hz'][0]:.6g} Hz | "
                    f"{details['matrix_policy']}"
                )
        return
    if not args.plots_only:
        for spec in groups:
            group = group_key(
                spec["model"], spec["temperature_K"],
                spec["separation_mm"], spec["count"],
            )
            full_key = (group, "alternate_full")
            previous_full = completed.get(full_key)
            retry_unsettled = bool(
                previous_full
                and not previous_full.get(
                    "online_settling_confirmation", {}
                ).get("confirmed", False)
                and args.max_runtime
                > float(previous_full.get("actual_final_time_s", 0.0)) + 1.0
            )
            if full_key not in completed or args.force or retry_unsettled:
                print(
                    f"RUN {group} | alternate full 6-DOF | adaptive to "
                    f"{100*args.settling_fraction:g}% with "
                    f"{args.hold_time:g}-s hold",
                    flush=True,
                )
                K, C, details = local_matrices(
                    spec["model"], spec["temperature_K"],
                    spec["separation_mm"], "alternate_full",
                )
                monitor = SettlementMonitor(
                    adjacent_pairs(spec["separation_mm"]),
                    args.settling_fraction, args.hold_time,
                    IMPULSE_TIME_S, sim.DYN.impulse_duration,
                )
                row = run_transient(
                    output=output, group=group,
                    model_name=spec["model"],
                    temperature_k=spec["temperature_K"],
                    separation_mm=spec["separation_mm"],
                    count=spec["count"],
                    representation="alternate_full",
                    local_K=K, local_C=C, matrix_details=details,
                    run_time_s=args.max_runtime,
                    timestep_s=args.timestep, log_stride=args.log_stride,
                    monitor=monitor,
                    force=(args.force or retry_unsettled),
                )
                completed[full_key] = row
                rows = list(completed.values()); write_rows(summary_path, rows)
            full = completed[full_key]
            confirmation = full.get("online_settling_confirmation", {})
            if not confirmation.get("confirmed", False):
                print(
                    f"STOP {group}: full 6-DOF did not remain below the "
                    f"threshold by {full['actual_final_time_s']:.3f} s. "
                    "Increase --max-runtime and rerun the same command.",
                    flush=True,
                )
                continue
            common_runtime = float(full["actual_final_time_s"])
            print(
                f"  full 6-DOF confirmed at {common_runtime:.3f} s; "
                "running matched comparison histories",
                flush=True,
            )
            for representation in ("primary", "alternate_translation"):
                key = (group, representation)
                if key in completed and not args.force:
                    print(f"SKIP {group} | {representation}", flush=True)
                    continue
                K, C, details = local_matrices(
                    spec["model"], spec["temperature_K"],
                    spec["separation_mm"], representation,
                )
                print(
                    f"RUN {group} | {representation} | "
                    f"matched duration {common_runtime:.3f} s",
                    flush=True,
                )
                completed[key] = run_transient(
                    output=output, group=group,
                    model_name=spec["model"],
                    temperature_k=spec["temperature_K"],
                    separation_mm=spec["separation_mm"],
                    count=spec["count"],
                    representation=representation,
                    local_K=K, local_C=C, matrix_details=details,
                    run_time_s=common_runtime,
                    timestep_s=args.timestep, log_stride=args.log_stride,
                    monitor=None, force=args.force,
                )
                rows = list(completed.values()); write_rows(summary_path, rows)
    rows = list(completed.values())
    write_rows(summary_path, rows)
    for spec in groups:
        plot_group(
            output,
            group_key(
                spec["model"], spec["temperature_K"],
                spec["separation_mm"], spec["count"],
            ),
            rows,
        )
    print(f"Results and plots saved to {output}")


if __name__ == "__main__":
    main()
