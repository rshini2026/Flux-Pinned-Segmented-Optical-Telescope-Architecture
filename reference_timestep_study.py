"""The default
4 s window contains
the recorded reference-case peaks; use --duration 16 for full-window checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import segment_dynamics as sim
from vertical_placement_study import adjacent_pairs

METRICS = (
    "mean_position_m", "mean_rotation_rad", "worst_position_m",
    "worst_rotation_rad", "adjacent_position_m", "adjacent_rotation_rad",
)


def evaluate(dt: float, duration: float) -> dict:
    sim.GEOM.gap = 0.010
    sim.GEOM.mechanisms_per_edge = 4
    sim.GEOM.row_separation_m = 0.045
    sim.GEOM.mechanism_edge_half_span_fraction = 0.45
    sim.DYN.zeta_nominal = 0.03
    sim.DYN.anchor_k_trans = 500.0
    sim.DYN.anchor_k_rot = 500.0
    sim.DYN.anchor_zeta_trans = 0.01
    sim.DYN.anchor_zeta_rot = 0.01
    sim.DYN.run_time_s = duration
    sim.DYN.impulse_time = 1.5
    sim.DYN.impulse_duration = 0.02
    sim.DYN.impulse_force = (0.25, 0.0, 0.15)
    sim.DYN.impulse_point_offset = (0.10, 0.15, 0.10)

    result = sim.run_case(temperature_k=77.0, temperature_model="chiang",
                          timestep=dt, show_viewer=False)
    t = np.asarray(result.t, dtype=float)
    q = np.asarray(result.segment_dofs, dtype=float)
    positions = sim.hex_positions(1, 0.5, 0.010)
    pairs = adjacent_pairs(positions, 0.010)
    pos = np.linalg.norm(q[:, :, :3], axis=2)
    rot = np.linalg.norm(q[:, :, 3:], axis=2)
    curves = {
        "mean_position_m": np.mean(pos, axis=1),
        "mean_rotation_rad": np.mean(rot, axis=1),
        "worst_position_m": np.max(pos, axis=1),
        "worst_rotation_rad": np.max(rot, axis=1),
        "adjacent_position_m": np.maximum.reduce([
            np.linalg.norm(q[:, i, :3] - q[:, j, :3], axis=1)
            for i, j in pairs]),
        "adjacent_rotation_rad": np.maximum.reduce([
            np.linalg.norm(q[:, i, 3:] - q[:, j, 3:], axis=1)
            for i, j in pairs]),
    }
    post = t >= sim.DYN.impulse_time
    row = {"dt_s": dt, "duration_s": duration,
           "last_sample_s": float(t[-1]), "force_clip_count": result.clip_count,
           "maximum_local_translation_m": sim.DYN.max_local_translation_m,
           "maximum_normal_gap_excursion_m": sim.DYN.max_normal_gap_excursion_m}
    for key, series in curves.items():
        indices = np.flatnonzero(post)
        k = int(indices[np.argmax(series[post])])
        row[key + "_peak"] = float(series[k])
        row[key + "_peak_time_s"] = float(t[k])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--steps", type=float, nargs="+",
                        default=(2e-4, 1e-4, 5e-5))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reference_timestep_study.json")
    args = parser.parse_args()
    if args.duration <= 1.52 or any(x <= 0 for x in args.steps):
        parser.error("Duration must exceed the pulse end and steps must be positive.")
    if any(abs((1.5 / d) - round(1.5 / d)) > 1e-7 or
           abs((1.52 / d) - round(1.52 / d)) > 1e-7 for d in args.steps):
        parser.error("Each timestep must exactly resolve both pulse boundaries.")
    rows = [evaluate(float(dt), args.duration) for dt in args.steps]
    fine = min(rows, key=lambda row: row["dt_s"])
    for row in rows:
        for metric in METRICS:
            value = row[metric + "_peak"]
            target = fine[metric + "_peak"]
            row[metric + "_peak_difference_pct"] = (
                100.0 * abs(value - target) / target if target else 0.0)
            row[metric + "_peak_time_difference_s"] = abs(
                row[metric + "_peak_time_s"] - fine[metric + "_peak_time_s"])
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"dt={row['dt_s']:.7f} s; max peak difference="
              f"{max(row[k + '_peak_difference_pct'] for k in METRICS):.6g}% ; "
              f"force clips={row['force_clip_count']}")
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
