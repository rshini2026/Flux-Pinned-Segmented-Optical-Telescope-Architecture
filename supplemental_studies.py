"""Run resumable supplemental simulations required by the final thesis plots.

Each completed case is stored immediately. Re-running the same command skips
completed cases, while ``--force`` intentionally recomputes them. Stiffness
and damping are selected from the field-cooling separation and remain fixed
throughout each transient.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import eigh

import segment_dynamics as sim
import vertical_placement_study as vertical
from thesis_config import (
    BASELINE_ALONG_EDGE_HALF_SPAN_MM,
    BASELINE_ROW_HALF_HEIGHT_MM,
    CENTER_RESTRAINT_DAMPING_RATIO,
    DEFAULT_LONG_TIME_S,
    DEFAULT_TIMESTEP_S,
    DEFAULT_TRANSIENT_TIME_S,
    IMPULSE_TIME_S,
    MECHANISM_DAMPING_RATIO,
    SUPPLEMENTAL_DIR,
    ensure_output_directories,
)
from mechanism_parameters import CONFIG, mechanism_C6, mechanism_K6


RESULTS_FILE = "supplemental_results.json"
MODAL_ASSEMBLY_SCHEMA = "full_local_rotational_terms"


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def case_key(case: dict) -> str:
    return str(case["key"])


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["key"]): row for row in rows}


def save_rows(path: Path, rows: dict[str, dict]) -> None:
    ordered = sorted(rows.values(), key=lambda row: str(row["key"]))
    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def adjacent_histories(dofs: np.ndarray, separation_mm: float) -> tuple[np.ndarray, np.ndarray]:
    positions = sim.hex_positions(1, sim.GEOM.hex_radius, separation_mm * 1e-3)
    pairs = vertical.adjacent_pairs(positions, separation_mm * 1e-3)
    rel_pos = np.zeros(dofs.shape[0])
    rel_rot = np.zeros(dofs.shape[0])
    for i, j in pairs:
        rel_pos = np.maximum(rel_pos, np.linalg.norm(dofs[:, i, :3] - dofs[:, j, :3], axis=1))
        rel_rot = np.maximum(rel_rot, np.linalg.norm(dofs[:, i, 3:] - dofs[:, j, 3:], axis=1))
    return rel_pos, rel_rot


def modal_from_matrices(K: np.ndarray, C: np.ndarray, M: np.ndarray) -> dict:
    values, vectors = eigh(0.5 * (K + K.T), M)
    values = np.asarray(values, dtype=float)
    frequencies = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    phi = vectors[:, 0]
    omega = math.sqrt(max(float(values[0]), 0.0))
    modal_mass = float(phi @ M @ phi)
    modal_c = float(phi @ C @ phi)
    zeta = modal_c / (2.0 * omega * modal_mass) if omega > 0 and modal_mass > 0 else None
    return {
        "first_mode_hz": float(frequencies[0]),
        "first_mode_damping_ratio": None if zeta is None else float(zeta),
        "first_six_frequencies_hz": frequencies[:6].tolist(),
    }


def primary_modal(case: dict) -> dict:
    gap_m = float(case["field_cooling_separation_mm"]) * 1e-3
    temperature = float(case["temperature_K"])
    model = str(case["model"])
    local_K = mechanism_K6(gap_m, temperature, model)
    local_C = mechanism_C6(
        gap_m, temperature, model,
        zeta=float(case.get("mechanism_zeta", MECHANISM_DAMPING_RATIO)),
    )
    K, C, M = assemble_with_local_matrices(case, local_K, local_C)
    return modal_from_matrices(K, C, M)


def assemble_with_local_matrices(case: dict, K6: np.ndarray, C6: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_k = vertical.mechanism_K6
    original_c = vertical.mechanism_C6
    vertical.mechanism_K6 = lambda *_args, **_kwargs: np.asarray(K6, dtype=float)
    vertical.mechanism_C6 = lambda *_args, **_kwargs: np.asarray(C6, dtype=float)
    try:
        return vertical.build_linear_system(
            float(case["temperature_K"]), str(case["model"]), int(case["count"]),
            float(case["field_cooling_separation_mm"]),
            float(case.get("row_half_height_mm", BASELINE_ROW_HALF_HEIGHT_MM)),
            float(case.get("half_span_mm", BASELINE_ALONG_EDGE_HALF_SPAN_MM)),
        )
    finally:
        vertical.mechanism_K6 = original_k
        vertical.mechanism_C6 = original_c


def run_transient(case: dict, output_dir: Path, timestep: float,
                  force: bool = False) -> dict:
    key = case_key(case)
    history_name = f"history_{key}.npz"
    history_path = output_dir / history_name
    if history_path.exists() and not force:
        with np.load(history_path) as saved:
            return json.loads(str(saved["metadata_json"].item()))

    old = {
        "gap": sim.GEOM.gap,
        "count": sim.GEOM.mechanisms_per_edge,
        "row": sim.GEOM.row_separation_m,
        "half": sim.GEOM.mechanism_edge_half_span_fraction,
        "runtime": sim.DYN.run_time_s,
        "zeta": sim.DYN.zeta_nominal,
        "anchor_t": sim.DYN.anchor_zeta_trans,
        "anchor_r": sim.DYN.anchor_zeta_rot,
        "min_gap": sim.DYN.separation_min_m,
    }
    sim.GEOM.gap = float(case["field_cooling_separation_mm"]) * 1e-3
    sim.GEOM.mechanisms_per_edge = int(case["count"])
    sim.GEOM.row_separation_m = 2.0 * float(case.get("row_half_height_mm", BASELINE_ROW_HALF_HEIGHT_MM)) * 1e-3
    sim.GEOM.mechanism_edge_half_span_fraction = float(case.get("half_span_mm", BASELINE_ALONG_EDGE_HALF_SPAN_MM)) / (sim.GEOM.hex_radius * 1e3)
    sim.DYN.run_time_s = float(case["runtime_s"])
    sim.DYN.zeta_nominal = float(case.get("mechanism_zeta", MECHANISM_DAMPING_RATIO))
    sim.DYN.anchor_zeta_trans = CENTER_RESTRAINT_DAMPING_RATIO
    sim.DYN.anchor_zeta_rot = CENTER_RESTRAINT_DAMPING_RATIO
    if float(case["field_cooling_separation_mm"]) < 5.0:
        sim.DYN.separation_min_m = 0.0001

    original_k: Callable = sim.mechanism_K6
    original_c: Callable = sim.mechanism_C6
    local_K6 = case.get("local_K6")
    local_C6 = case.get("local_C6")
    if local_K6 is not None:
        fixed_k = np.asarray(local_K6, dtype=float)
        fixed_c = np.asarray(local_C6, dtype=float)
        sim.mechanism_K6 = lambda *_args, **_kwargs: fixed_k
        sim.mechanism_C6 = lambda *_args, **_kwargs: fixed_c

    started = time.perf_counter()
    try:
        result = sim.run_case(
            temperature_k=float(case["temperature_K"]),
            temperature_model=str(case["model"]),
            show_viewer=False,
            timestep=float(timestep),
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
        sim.DYN.separation_min_m = old["min_gap"]

    t = np.asarray(result.t, dtype=float)
    dofs = np.asarray(result.segment_dofs, dtype=float)
    segment_pos = np.linalg.norm(dofs[:, :, :3], axis=2)
    segment_rot = np.linalg.norm(dofs[:, :, 3:], axis=2)
    relative_pos, relative_rot = adjacent_histories(
        dofs, float(case["field_cooling_separation_mm"])
    )
    histories = {
        "t": t,
        "mean_position_m": np.asarray(result.pos_err, dtype=float),
        "mean_rotation_rad": np.asarray(result.rot_err, dtype=float),
        "worst_position_m": np.max(segment_pos, axis=1),
        "worst_rotation_rad": np.max(segment_rot, axis=1),
        "adjacent_position_m": relative_pos,
        "adjacent_rotation_rad": relative_rot,
    }
    post = t >= IMPULSE_TIME_S
    metadata = {
        **{k: v for k, v in case.items() if k not in {"local_K6", "local_C6"}},
        "complete": True,
        "history_file": history_name,
        "wall_runtime_s": float(time.perf_counter() - started),
        "center_restraint_damping_ratio": CENTER_RESTRAINT_DAMPING_RATIO,
        "field_cooling_matrices_fixed_during_transient": True,
        "max_mechanism_force_N": float(result.max_force),
        "max_mechanism_torque_Nm": float(result.max_torque),
        "force_clip_count": int(result.clip_count),
    }
    for name, values in histories.items():
        if name == "t":
            continue
        metadata[f"peak_{name}"] = float(np.max(values[post]))
    np.savez_compressed(history_path, **histories,
                        metadata_json=np.array(json.dumps(metadata)))
    return metadata


def cases(long_runtime: float) -> list[dict]:
    base = {
        "model": "chiang", "temperature_K": 77.0,
        "field_cooling_separation_mm": 10.0, "count": 4,
        "row_half_height_mm": 22.5, "half_span_mm": 225.0,
        "mechanism_zeta": 0.03,
    }
    planned: list[dict] = [
        {**base, "key": "reference", "study": "reference", "runtime_s": 16.0},
    ]
    for zeta in (0.02, 0.03, 0.04):
        planned.append({**base, "key": f"damping_zeta_{slug(zeta)}",
                        "study": "damping", "mechanism_zeta": zeta,
                        "runtime_s": long_runtime})
    planned.extend([
        {**base, "key": "settling_reference", "study": "settling",
         "runtime_s": long_runtime},
        {**base, "key": "settling_best", "study": "settling",
         "temperature_K": 50.0, "field_cooling_separation_mm": 7.5,
         "count": 10, "runtime_s": long_runtime},
        {**base, "key": "extrapolated_best", "study": "extrapolated",
         "temperature_K": 50.0, "field_cooling_separation_mm": 7.5,
         "count": 10, "runtime_s": 30.0},
        {**base, "key": "extrapolated_40K_1mm", "study": "extrapolated",
         "temperature_K": 40.0, "field_cooling_separation_mm": 1.0,
         "count": 10, "runtime_s": 30.0},
    ])
    for half_span in (150.0, 200.0, 225.0):
        planned.append({**base, "key": f"along_edge_{slug(half_span)}mm",
                        "study": "along_edge", "temperature_K": 50.0,
                        "field_cooling_separation_mm": 7.5, "count": 10,
                        "half_span_mm": half_span, "runtime_s": 16.0})

    primary_K = mechanism_K6(0.025, 77.0, "chiang")
    primary_C = mechanism_C6(0.025, 77.0, "chiang", zeta=0.03)
    alternate_K_translation = np.diag([5.40, 8.77, 5.40, 0.0, 0.0, 0.0])
    alternate_C_translation = np.diag([0.0113, 0.0149, 0.0113, 0.0, 0.0, 0.0])
    alternate_K_full = np.diag([5.40, 8.77, 5.40, 292.0, 0.0, 358.0])
    alternate_C_full = np.diag([0.0113, 0.0149, 0.0113, 0.4306, 0.0, 0.4042])
    for key, label, K, C in (
        ("alternate_primary", "Primary model", primary_K, primary_C),
        ("alternate_translation", "Alternate translations only", alternate_K_translation, alternate_C_translation),
        ("alternate_full", "Alternate full 6-DOF matrix", alternate_K_full, alternate_C_full),
    ):
        planned.append({**base, "key": key, "study": "alternate_matrix",
                        "label": label, "field_cooling_separation_mm": 25.0,
                        "runtime_s": 16.0, "local_K6": K.tolist(),
                        "local_C6": C.tolist()})
    return planned


def add_modal(case: dict) -> dict:
    if case.get("local_K6") is None:
        return primary_modal(case)
    K, C, M = assemble_with_local_matrices(
        case, np.asarray(case["local_K6"]), np.asarray(case["local_C6"])
    )
    return modal_from_matrices(K, C, M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SUPPLEMENTAL_DIR)
    parser.add_argument("--studies", nargs="+", choices=(
        "reference", "damping", "settling", "extrapolated",
        "along_edge", "alternate_matrix", "all"), default=["all"])
    parser.add_argument("--long-runtime", type=float, default=DEFAULT_LONG_TIME_S)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP_S)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_directories()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULTS_FILE
    rows = load_rows(result_path)
    selected = set(args.studies)
    plan = [case for case in cases(float(args.long_runtime))
            if "all" in selected or case["study"] in selected]
    print(f"Supplemental cases: {len(plan)}")
    completed_now = 0
    for index, case in enumerate(plan, 1):
        key = case_key(case)
        if key in rows and rows[key].get("complete") and not args.force:
            if rows[key].get("modal_assembly_schema") != MODAL_ASSEMBLY_SCHEMA:
                rows[key]["modal"] = add_modal(case)
                rows[key]["modal_assembly_schema"] = MODAL_ASSEMBLY_SCHEMA
                save_rows(result_path, rows)
                print(f"[{index}/{len(plan)}] REFRESH MODAL {key}")
            else:
                print(f"[{index}/{len(plan)}] SKIP {key}")
            continue
        if args.max_cases is not None and completed_now >= args.max_cases:
            break
        print(f"[{index}/{len(plan)}] RUN {key}")
        result = run_transient(case, output, float(args.timestep), force=args.force)
        result["modal"] = add_modal(case)
        result["modal_assembly_schema"] = MODAL_ASSEMBLY_SCHEMA
        rows[key] = result
        save_rows(result_path, rows)
        completed_now += 1
        print(f"    complete in {result['wall_runtime_s']:.1f} s")
    save_rows(result_path, rows)
    print(f"Saved {len(rows)} completed cases to {result_path}")


if __name__ == "__main__":
    main()
