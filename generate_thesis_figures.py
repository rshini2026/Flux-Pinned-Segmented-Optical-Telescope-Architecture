"""Create the final numbered thesis figures from completed checkpoint data.

This plotting pass performs no simulations and may be run at any time. Figures
whose required cases are not yet complete are skipped and listed in the
manifest.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_config import (
    FIGURE_DIR, FIELD_COOLING_SEPARATIONS_MM, MAIN_SWEEP_DIR,
    MECHANISM_COUNTS, SUPPLEMENTAL_DIR, TEMPERATURES_K, VERTICAL_DIR,
    ensure_output_directories,
)
from empirical_mechanism_data import (
    CHIANG_FORCE_N, CHIANG_T_K, JIANG_FORCE_N, JIANG_T_K,
    jiang_pchip_force_n, temperature_pchip_ratio,
)
from mechanism_parameters import mechanism_K6, temperature_ratio


plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.titlesize": 15,
    "savefig.dpi": 300,
})


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, output: Path, filename: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main_rows(directory: Path) -> list[dict]:
    return load_json(directory / "main_sweep_z22p5mm.json", [])


def complete(row: dict, dynamic: bool) -> bool:
    return bool(row.get("dynamic_complete" if dynamic else "modal_complete", False))


def match(rows: list[dict], model: str, temperature: float,
          separation: float, count: int, dynamic: bool = False) -> dict | None:
    for row in rows:
        if (str(row.get("model", "")).lower() == model
                and math.isclose(float(row.get("temperature_K", -1)), temperature)
                and math.isclose(float(row.get("separation_mm", -1)), separation)
                and int(row.get("mechanism_count_per_interface", -1)) == count
                and complete(row, dynamic)):
            return row
    return None


def metric_matrix(rows: list[dict], model: str, count: int, field: str,
                  scale: float = 1.0, dynamic: bool = False) -> np.ndarray | None:
    values = np.full((len(TEMPERATURES_K), len(FIELD_COOLING_SEPARATIONS_MM)), np.nan)
    for i, temperature in enumerate(TEMPERATURES_K):
        for j, separation in enumerate(FIELD_COOLING_SEPARATIONS_MM):
            row = match(rows, model, temperature, separation, count, dynamic)
            if row is not None and row.get(field) is not None:
                values[i, j] = float(row[field]) * scale
    return values if np.isfinite(values).all() else None


def annotate(ax: plt.Axes, values: np.ndarray, limit: float | None = None,
             fmt: str = ".3g") -> None:
    if limit is None:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        midpoint, span = 0.5 * (lo + hi), max(hi - lo, 1e-15)
        is_dark = lambda value: abs(value - midpoint) > 0.30 * span
    else:
        is_dark = lambda value: abs(value) > 0.52 * limit
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        color="white" if is_dark(values[i, j]) else "black", fontsize=8.5)


def decorate_heatmap(ax: plt.Axes, values: np.ndarray, vmin: float, vmax: float,
                     cmap: str = "viridis"):
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(FIELD_COOLING_SEPARATIONS_MM)),
                  labels=[f"{x:g}" for x in FIELD_COOLING_SEPARATIONS_MM])
    ax.set_yticks(range(len(TEMPERATURES_K)),
                  labels=[f"{x:g}" for x in TEMPERATURES_K])
    ax.set_xlabel("Field-cooling separation (mm)")
    return image


def figure_4_5(output: Path) -> Path:
    t = np.linspace(50.0, 85.0, 400)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for model, color in (("chiang", "C0"), ("jiang", "C1")):
        ax.plot(t, [temperature_ratio(x, model) for x in t], color=color,
                linewidth=2.2, label=model.title())
    chiang_mask = (CHIANG_T_K >= 50) & (CHIANG_T_K <= 85)
    ax.scatter(CHIANG_T_K[chiang_mask],
               temperature_pchip_ratio(CHIANG_T_K[chiang_mask], 77.0,
                                       CHIANG_T_K, CHIANG_FORCE_N),
               color="C0", marker="o", zorder=3)
    jiang_mask = (JIANG_T_K >= 50) & (JIANG_T_K <= 85)
    jiang_ref = float(jiang_pchip_force_n(77.0))
    ax.scatter(JIANG_T_K[jiang_mask], JIANG_FORCE_N[jiang_mask] / jiang_ref,
               color="C1", marker="s", zorder=3)
    ax.axvline(77.0, color="0.35", linestyle=":", label="77 K reference")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalized stiffness multiplier")
    ax.set_title("Temperature Scaling Models Normalized at 77 K")
    ax.grid(alpha=0.28); ax.legend()
    return save(fig, output, "figure_4_5_temperature_scaling_models.png")


def figure_4_6(output: Path) -> Path:
    d = np.linspace(5.0, 30.0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for temperature in TEMPERATURES_K:
        kp, kn = [], []
        for separation in d:
            K = mechanism_K6(separation * 1e-3, temperature, "chiang")
            kp.append(K[0, 0]); kn.append(K[1, 1])
        axes[0].plot(d, kp, linewidth=2, label=f"{temperature:g} K")
        axes[1].plot(d, kn, linewidth=2, label=f"{temperature:g} K")
    axes[0].set_title("Parallel stiffness")
    axes[1].set_title("Perpendicular stiffness")
    for ax in axes:
        ax.set_xlabel("Field-cooling separation (mm)")
        ax.set_ylabel("Stiffness per mechanism (N/m)")
        ax.grid(alpha=0.28); ax.legend()
    fig.suptitle("Directional Flux-Pinning Stiffness Model")
    return save(fig, output, "figure_4_6_directional_stiffness_model.png")


def heatmap_pair(rows: list[dict], output: Path, filename: str, title: str,
                 field: str, label: str, scale: float, dynamic: bool) -> Path | None:
    matrices = [metric_matrix(rows, model, 4, field, scale, dynamic) for model in ("chiang", "jiang")]
    if any(values is None for values in matrices):
        return None
    finite = np.concatenate([values.ravel() for values in matrices])
    vmin, vmax = float(finite.min()), float(finite.max())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True, constrained_layout=True)
    for ax, model, values in zip(axes, ("chiang", "jiang"), matrices):
        image = decorate_heatmap(ax, values, vmin, vmax)
        ax.set_title(model.title()); annotate(ax, values)
    axes[0].set_ylabel("Temperature (K)")
    fig.colorbar(image, ax=axes, label=label, shrink=0.83)
    fig.suptitle(title + "\nN = 4, z = ±22.5 mm, s = ±225 mm")
    return save(fig, output, filename)


def two_metric_grid(rows: list[dict], output: Path, filename: str, title: str,
                    metrics: list[tuple[str, float, str]]) -> Path | None:
    data = []
    for field, scale, label in metrics:
        matrices = [metric_matrix(rows, model, 4, field, scale, True)
                    for model in ("chiang", "jiang")]
        if any(values is None for values in matrices):
            return None
        data.append((label, matrices))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for r, (label, matrices) in enumerate(data):
        finite = np.concatenate([values.ravel() for values in matrices])
        vmin, vmax = float(finite.min()), float(finite.max())
        for c, (model, values) in enumerate(zip(("chiang", "jiang"), matrices)):
            image = decorate_heatmap(axes[r, c], values, vmin, vmax)
            axes[r, c].set_title(f"{model.title()} — {label}"); annotate(axes[r, c], values)
        axes[r, 0].set_ylabel("Temperature (K)")
        fig.colorbar(image, ax=axes[r, :], label=label, shrink=0.82)
    fig.suptitle(title + "\nN = 4, z = ±22.5 mm, s = ±225 mm")
    return save(fig, output, filename)


def figure_5_5(rows: list[dict], output: Path) -> Path | None:
    specs = [
        ("first_mode_hz", 1.0, "Fundamental frequency", False),
        ("peak_mean_position_error_m", 1e3, "Peak mean position", True),
        ("peak_mean_rotation_error_rad", 1e3, "Peak mean rotation", True),
        ("peak_adjacent_relative_position_error_m", 1e3, "Adjacent position", True),
        ("peak_adjacent_relative_rotation_error_rad", 1e3, "Adjacent rotation", True),
    ]
    panels = []
    for field, scale, title, dynamic in specs:
        a = metric_matrix(rows, "chiang", 4, field, scale, dynamic)
        b = metric_matrix(rows, "jiang", 4, field, scale, dynamic)
        if a is None or b is None:
            return None
        panels.append((title, 100.0 * (b - a) / a))
    limit = max(float(np.max(np.abs(values))) for _, values in panels)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), sharex=True, sharey=True,
                             constrained_layout=True)
    used = []
    for ax, (title, values) in zip(axes.flat, panels):
        used.append(ax)
        image = decorate_heatmap(ax, values, -limit, limit, "coolwarm")
        ax.set_title(title); annotate(ax, values, limit, "+.1f")
    axes[0, 0].set_ylabel("Temperature (K)"); axes[1, 0].set_ylabel("Temperature (K)")
    axes[1, 2].axis("off")
    fig.colorbar(image, ax=used, label="Jiang change relative to Chiang (%)", shrink=0.82)
    fig.suptitle("Chiang–Jiang Structural Response Comparison\nN = 4, z = ±22.5 mm, s = ±225 mm")
    return save(fig, output, "figure_5_5_chiang_jiang_percent_change.png")


def supplemental_rows(directory: Path) -> dict[str, dict]:
    rows = load_json(directory / "supplemental_results.json", [])
    return {str(row["key"]): row for row in rows if row.get("complete")}


def history(directory: Path, row: dict) -> dict[str, np.ndarray] | None:
    path = directory / str(row.get("history_file", ""))
    if not path.exists():
        return None
    with np.load(path) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files
                if name != "metadata_json"}


def standard_history_figure(output: Path, filename: str, title: str,
                            histories: list[tuple[str, dict]], window: float | None = None) -> Path:
    panels = (
        ("mean_position_m", 1e3, "Mean position error (mm)"),
        ("mean_rotation_rad", 1e3, "Mean rotation error (mrad)"),
        ("adjacent_position_m", 1e3, "Maximum adjacent position error (mm)"),
        ("adjacent_rotation_rad", 1e3, "Maximum adjacent rotation error (mrad)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, constrained_layout=True)
    for label, values in histories:
        t = values["t"]
        mask = np.ones(len(t), dtype=bool) if window is None else t <= window
        for ax, (key, scale, ylabel) in zip(axes.flat, panels):
            ax.plot(t[mask], values[key][mask] * scale, linewidth=1.25, label=label)
            ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
            ax.axvline(1.5, color="0.35", linestyle=":", linewidth=1)
    for ax in axes[1]: ax.set_xlabel("Time (s)")
    axes[0, 0].legend()
    fig.suptitle(title)
    return save(fig, output, filename)


def figure_5_1(srows: dict[str, dict], supplemental: Path, output: Path) -> Path | None:
    row = srows.get("reference")
    h = history(supplemental, row) if row else None
    if h is None: return None
    return standard_history_figure(output, "figure_5_1_reference_impulse_response.png",
        "Reference Structural Impulse Response\nChiang, 77 K, 10 mm field-cooling separation, N = 4",
        [("Response", h)], window=16.0)


def period_envelope(t: np.ndarray, values: np.ndarray, impulse_time: float,
                    frequency: float) -> tuple[np.ndarray, np.ndarray]:
    period = 1.0 / frequency
    x, y, start = [], [], impulse_time
    while start < t[-1]:
        stop = min(start + period, float(t[-1]) + 1e-12)
        mask = (t >= start) & (t < stop)
        if not np.any(mask): break
        x.append(0.5 * (start + stop) - impulse_time)
        y.append(float(np.max(values[mask])))
        start += period
    y = np.asarray(y)
    if len(y) and y[0] > 0: y = y / y[0]
    return np.asarray(x), y


def figure_5_6(srows: dict[str, dict], supplemental: Path, output: Path) -> Path | None:
    rows = [srows.get(f"damping_zeta_{str(z).replace('.', 'p')}") for z in (0.02, 0.03, 0.04)]
    if any(row is None for row in rows): return None
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True, constrained_layout=True)
    for zeta, row in zip((0.02, 0.03, 0.04), rows):
        h = history(supplemental, row)
        if h is None: return None
        frequency = float(row["modal"]["first_mode_hz"])
        for ax, key in zip(axes, ("mean_position_m", "mean_rotation_rad")):
            x, y = period_envelope(h["t"], h[key], 1.5, frequency)
            ax.plot(x, y, marker="o", markersize=3, label=f"ζ = {zeta:.02f}")
    axes[0].set_ylabel("Normalized position envelope")
    axes[1].set_ylabel("Normalized rotation envelope"); axes[1].set_xlabel("Time after impulse (s)")
    for ax in axes:
        ax.axhline(0.02, color="0.35", linestyle=":", label="2% threshold")
        ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Mechanism-Damping Sensitivity\nChiang, 77 K, 10 mm field-cooling separation, N = 4")
    return save(fig, output, "figure_5_6_mechanism_damping_sensitivity.png")


def figure_5_7(srows: dict[str, dict], supplemental: Path, output: Path) -> Path | None:
    specs = (("settling_reference", "Reference: 77 K, 10 mm, N = 4"),
             ("settling_best", "Best evaluated: 50 K, 7.5 mm, N = 10"))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True, constrained_layout=True)
    for key, label in specs:
        row = srows.get(key); h = history(supplemental, row) if row else None
        if h is None: plt.close(fig); return None
        f1 = float(row["modal"]["first_mode_hz"])
        for ax, series in zip(axes, ("mean_position_m", "mean_rotation_rad")):
            x, y = period_envelope(h["t"], h[series], 1.5, f1)
            ax.plot(x, y, linewidth=1.8, label=label)
    axes[0].set_ylabel("Normalized position envelope")
    axes[1].set_ylabel("Normalized rotation envelope"); axes[1].set_xlabel("Time after impulse (s)")
    for ax in axes:
        ax.axhline(0.02, color="0.35", linestyle=":", label="2% threshold")
        ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Long-Duration Settling Comparison — Chiang Model")
    return save(fig, output, "figure_5_7_long_duration_settling.png")


def figure_5_8_9(srows: dict[str, dict], supplemental: Path, output: Path) -> list[Path]:
    a, b = srows.get("extrapolated_best"), srows.get("extrapolated_40K_1mm")
    ha, hb = (history(supplemental, a) if a else None,
              history(supplemental, b) if b else None)
    if ha is None or hb is None: return []
    out = [standard_history_figure(output, "figure_5_8_extrapolated_histories.png",
        "Evaluated Best Case and Extrapolated Prediction — Chiang Model\nN = 10, z = ±22.5 mm, s = ±225 mm",
        [("50 K, 7.5 mm", ha), ("40 K, 1 mm (extrapolated)", hb)], window=16.0)]
    panels = [
        ("Fundamental frequency", float(a["modal"]["first_mode_hz"]), float(b["modal"]["first_mode_hz"]), "Hz"),
        ("Peak mean position", a["peak_mean_position_m"]*1e3, b["peak_mean_position_m"]*1e3, "mm"),
        ("Peak mean rotation", a["peak_mean_rotation_rad"]*1e3, b["peak_mean_rotation_rad"]*1e3, "mrad"),
        ("Peak adjacent position", a["peak_adjacent_position_m"]*1e3, b["peak_adjacent_position_m"]*1e3, "mm"),
        ("Peak adjacent rotation", a["peak_adjacent_rotation_rad"]*1e3, b["peak_adjacent_rotation_rad"]*1e3, "mrad"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for ax, (title, va, vb, unit) in zip(axes.flat, panels):
        bars = ax.bar(["50 K, 7.5 mm", "40 K, 1 mm"], [va, vb], color=["C0", "C1"])
        ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.4g", padding=3)
        ax.text(0.5, 0.93, f"Change: {100*(vb-va)/va:+.1f}%", transform=ax.transAxes,
                ha="center", va="top")
    axes[1, 2].axis("off")
    fig.suptitle("Predicted Structural Response at 40 K and 1 mm Separation\nChiang model; 1 mm separation is extrapolated")
    out.append(save(fig, output, "figure_5_9_extrapolated_metrics.png"))
    return out


def count_trend(rows: list[dict], output: Path, filename: str, title: str,
                specs: list[tuple[str, float, str]]) -> Path | None:
    fig, axes = plt.subplots(1, len(specs), figsize=(6.2*len(specs), 5.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (field, scale, ylabel) in zip(axes, specs):
        for model in ("chiang", "jiang"):
            values = []
            for count in MECHANISM_COUNTS:
                row = match(rows, model, 50.0, 7.5, count, field != "first_mode_hz")
                if row is None: plt.close(fig); return None
                values.append(float(row[field]) * scale)
            ax.plot(MECHANISM_COUNTS, values, marker="o", linewidth=2, label=model.title())
        ax.set_xlabel("Mechanisms per interface, N"); ax.set_ylabel(ylabel)
        ax.set_xticks(MECHANISM_COUNTS); ax.grid(alpha=0.28); ax.legend()
    fig.suptitle(title + "\n50 K, 7.5 mm field-cooling separation, z = ±22.5 mm")
    return save(fig, output, filename)


def count_heatmaps(rows: list[dict], output: Path, filename: str, title: str,
                   field: str, label: str, scale: float, dynamic: bool) -> Path | None:
    all_mats = []
    for model in ("chiang", "jiang"):
        mats = [metric_matrix(rows, model, count, field, scale, dynamic) for count in MECHANISM_COUNTS]
        if any(m is None for m in mats): return None
        all_mats.append(mats)
    finite = np.concatenate([m.ravel() for mats in all_mats for m in mats])
    vmin, vmax = float(finite.min()), float(finite.max())
    fig, axes = plt.subplots(2, len(MECHANISM_COUNTS), figsize=(17.5, 7.6), sharex=True,
                             sharey=True, constrained_layout=True)
    for r, model in enumerate(("chiang", "jiang")):
        for c, (count, values) in enumerate(zip(MECHANISM_COUNTS, all_mats[r])):
            image = decorate_heatmap(axes[r, c], values, vmin, vmax)
            axes[r, c].set_title(f"{model.title()}, N = {count}")
            annotate(axes[r, c], values)
        axes[r, 0].set_ylabel("Temperature (K)")
    fig.colorbar(image, ax=axes, label=label, shrink=0.82)
    fig.suptitle(title + "\nz = ±22.5 mm, s = ±225 mm")
    return save(fig, output, filename)


def figure_6_9(rows: list[dict], main_dir: Path, output: Path) -> Path | None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for count in MECHANISM_COUNTS:
        row = match(rows, "chiang", 50.0, 7.5, count, True)
        if row is None or not row.get("history_file"): plt.close(fig); return None
        path = main_dir / str(row["history_file"])
        if not path.exists(): plt.close(fig); return None
        with np.load(path) as h:
            axes[0].plot(h["t"], h["mean_position_error_m"]*1e3, label=f"N = {count}")
            axes[1].plot(h["t"], h["mean_rotation_error_rad"]*1e3, label=f"N = {count}")
    axes[0].set_ylabel("Mean position error (mm)")
    axes[1].set_ylabel("Mean rotation error (mrad)"); axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(1.5, color="0.35", linestyle=":"); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("Impulse Response versus Mechanism Count\nChiang, 50 K, 7.5 mm field-cooling separation")
    return save(fig, output, "figure_6_9_mechanism_count_histories.png")


def figure_6_11_12(srows: dict[str, dict], output: Path) -> list[Path]:
    rows = [srows.get(f"along_edge_{int(x)}mm") for x in (150, 200, 225)]
    if any(row is None for row in rows): return []
    x = np.array([300, 400, 450], dtype=float)
    outputs = []
    for filename, title, specs in (
        ("figure_6_11_along_edge_absolute_response.png", "Peak Structural Response versus Along-Edge Placement", (
            ("peak_mean_position_m", 1e3, "Mean position"),
            ("peak_worst_position_m", 1e3, "Worst-segment position"),
            ("peak_mean_rotation_rad", 1e3, "Mean rotation"),
            ("peak_worst_rotation_rad", 1e3, "Worst-segment rotation"))),
        ("figure_6_12_along_edge_adjacent_response.png", "Adjacent-Segment Response versus Along-Edge Placement", (
            ("peak_adjacent_position_m", 1e3, "Adjacent position"),
            ("peak_adjacent_rotation_rad", 1e3, "Adjacent rotation"))),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        position = [spec for spec in specs if "position" in spec[0]]
        rotation = [spec for spec in specs if "rotation" in spec[0]]
        for ax, group, ylabel in ((axes[0], position, "Peak position error (mm)"),
                                  (axes[1], rotation, "Peak rotation error (mrad)")):
            for field, scale, label in group:
                ax.plot(x, [float(row[field])*scale for row in rows], marker="o", label=label)
            ax.set_xlabel("Along-edge footprint (mm)"); ax.set_ylabel(ylabel)
            ax.set_xticks(x); ax.grid(alpha=0.28); ax.legend()
        fig.suptitle(title + "\nChiang, 50 K, 7.5 mm field-cooling separation, N = 10")
        outputs.append(save(fig, output, filename))
    return outputs


def vertical_figures(vertical_dir: Path, output: Path) -> list[Path]:
    rows = load_json(vertical_dir / "vertical_placement_results.json", [])
    rows = sorted([r for r in rows if r.get("modal_complete")], key=lambda r: float(r["row_half_height_z_mm"]))
    if len(rows) != 3: return []
    z = np.array([float(r["row_half_height_z_mm"]) for r in rows])
    outputs = []
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    ax.plot(z, [float(r["first_mode_hz"]) for r in rows], marker="o", linewidth=2)
    ax.set_xlabel("Mechanism-row half-height, |z| (mm)"); ax.set_ylabel("Fundamental frequency (Hz)")
    ax.set_xticks(z); ax.grid(alpha=0.28); ax.set_title("Fundamental Frequency versus Vertical Row Placement")
    outputs.append(save(fig, output, "figure_6_13_vertical_frequency.png"))
    if not all(r.get("dynamic_complete") for r in rows): return outputs
    for filename, title, specs in (
        ("figure_6_14_vertical_absolute_response.png", "Peak Structural Response versus Vertical Row Placement", (
            ("peak_mean_position_error_m", 1e3, "Assembly mean position"),
            ("peak_worst_segment_position_error_m", 1e3, "Worst-segment position"),
            ("peak_mean_rotation_error_rad", 1e3, "Assembly mean rotation"),
            ("peak_worst_segment_rotation_error_rad", 1e3, "Worst-segment rotation"))),
        ("figure_6_15_vertical_adjacent_response.png", "Adjacent-Segment Response versus Vertical Row Placement", (
            ("peak_adjacent_relative_position_error_m", 1e3, "Adjacent position"),
            ("peak_adjacent_relative_rotation_error_rad", 1e3, "Adjacent rotation"))),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, token, ylabel in ((axes[0], "position", "Peak position error (mm)"),
                                  (axes[1], "rotation", "Peak rotation error (mrad)")):
            for field, scale, label in specs:
                if token in field:
                    ax.plot(z, [float(r[field])*scale for r in rows], marker="o", label=label)
            ax.set_xlabel("Mechanism-row half-height, |z| (mm)"); ax.set_ylabel(ylabel)
            ax.set_xticks(z); ax.grid(alpha=0.28); ax.legend()
        fig.suptitle(title + "\nChiang, 50 K, 7.5 mm field-cooling separation, N = 10")
        outputs.append(save(fig, output, filename))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for row in rows:
        path = vertical_dir / str(row["history_file"])
        if not path.exists(): plt.close(fig); return outputs
        with np.load(path) as h:
            label = f"z = ±{float(row['row_half_height_z_mm']):g} mm"
            axes[0].plot(h["t"], h["mean_position_error_m"]*1e3, label=label)
            axes[1].plot(h["t"], h["mean_rotation_error_rad"]*1e3, label=label)
    axes[0].set_ylabel("Mean position error (mm)")
    axes[1].set_ylabel("Mean rotation error (mrad)"); axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(1.5, color="0.35", linestyle=":"); ax.grid(alpha=0.28); ax.legend()
    fig.suptitle("Structural Response Histories versus Vertical Row Placement\nChiang, 50 K, 7.5 mm field-cooling separation, N = 10")
    outputs.append(save(fig, output, "figure_6_16_vertical_histories.png"))
    return outputs


def figure_7_1_2(srows: dict[str, dict], supplemental: Path, output: Path) -> list[Path]:
    keys = ("alternate_primary", "alternate_translation", "alternate_full")
    rows = [srows.get(key) for key in keys]
    if any(row is None for row in rows): return []
    labels = [str(row["label"]) for row in rows]
    histories = []
    for label, row in zip(labels, rows):
        h = history(supplemental, row)
        if h is None: return []
        histories.append((label, h))
    outputs = [standard_history_figure(output, "figure_7_1_alternate_matrix_histories.png",
        "Alternate Matrix Structural Response Histories\n77 K, 25 mm field-cooling separation, N = 4",
        histories, window=16.0)]
    specs = (("peak_mean_position_m", 1e3, "Peak mean position (mm)"),
             ("peak_mean_rotation_rad", 1e3, "Peak mean rotation (mrad)"),
             ("peak_adjacent_position_m", 1e3, "Peak adjacent position (mm)"),
             ("peak_adjacent_rotation_rad", 1e3, "Peak adjacent rotation (mrad)"))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    colors = ["C0", "C1", "C3"]
    for ax, (field, scale, ylabel) in zip(axes.flat, specs):
        values = [float(row[field])*scale for row in rows]
        bars = ax.bar(labels, values, color=colors); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.3g", padding=3); ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Alternate Matrix Peak Structural Response\n77 K, 25 mm field-cooling separation, N = 4")
    outputs.append(save(fig, output, "figure_7_2_alternate_matrix_peak_response.png"))
    return outputs


def generate(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    ensure_output_directories()
    output = args.figure_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = main_rows(args.main_sweep_dir)
    srows = supplemental_rows(args.supplemental_dir)
    made: list[Path] = [figure_4_5(output), figure_4_6(output)]
    skipped: list[str] = []
    requests = [
        ("Figure 5.1", lambda: figure_5_1(srows, args.supplemental_dir, output)),
        ("Figure 5.2", lambda: heatmap_pair(rows, output, "figure_5_2_fundamental_frequency.png", "Fundamental Frequency Across Temperature and Separation", "first_mode_hz", "Fundamental frequency (Hz)", 1.0, False)),
        ("Figure 5.3", lambda: two_metric_grid(rows, output, "figure_5_3_peak_mean_response.png", "Peak Assembly-Mean Structural Response", [("peak_mean_position_error_m",1e3,"Peak mean position error (mm)"),("peak_mean_rotation_error_rad",1e3,"Peak mean rotation error (mrad)")])),
        ("Figure 5.4", lambda: two_metric_grid(rows, output, "figure_5_4_peak_adjacent_response.png", "Peak Adjacent-Segment Relative Response", [("peak_adjacent_relative_position_error_m",1e3,"Peak adjacent position error (mm)"),("peak_adjacent_relative_rotation_error_rad",1e3,"Peak adjacent rotation error (mrad)")])),
        ("Figure 5.5", lambda: figure_5_5(rows, output)),
        ("Figure 5.6", lambda: figure_5_6(srows, args.supplemental_dir, output)),
        ("Figure 5.7", lambda: figure_5_7(srows, args.supplemental_dir, output)),
        ("Figures 5.8–5.9", lambda: figure_5_8_9(srows, args.supplemental_dir, output)),
        ("Figure 6.1", lambda: count_trend(rows, output, "figure_6_1_frequency_vs_count.png", "Fundamental Frequency versus Mechanism Count", [("first_mode_hz",1.0,"Fundamental frequency (Hz)")])),
        ("Figure 6.2", lambda: count_heatmaps(rows, output, "figure_6_2_frequency_heatmaps.png", "Fundamental Frequency versus Mechanism Count", "first_mode_hz", "Fundamental frequency (Hz)",1.0,False)),
        ("Figure 6.3", lambda: count_trend(rows, output, "figure_6_3_absolute_response_vs_count.png", "Absolute Segment Response versus Mechanism Count", [("peak_mean_position_error_m",1e3,"Peak mean position error (mm)"),("peak_mean_rotation_error_rad",1e3,"Peak mean rotation error (mrad)")])),
        ("Figure 6.4", lambda: count_heatmaps(rows, output, "figure_6_4_mean_position_heatmaps.png", "Peak Mean Position Error versus Mechanism Count", "peak_mean_position_error_m", "Peak mean position error (mm)",1e3,True)),
        ("Figure 6.5", lambda: count_heatmaps(rows, output, "figure_6_5_mean_rotation_heatmaps.png", "Peak Mean Rotation Error versus Mechanism Count", "peak_mean_rotation_error_rad", "Peak mean rotation error (mrad)",1e3,True)),
        ("Figure 6.6", lambda: count_trend(rows, output, "figure_6_6_adjacent_response_vs_count.png", "Adjacent-Segment Response versus Mechanism Count", [("peak_adjacent_relative_position_error_m",1e3,"Peak adjacent position error (mm)"),("peak_adjacent_relative_rotation_error_rad",1e3,"Peak adjacent rotation error (mrad)")])),
        ("Figure 6.7", lambda: count_heatmaps(rows, output, "figure_6_7_adjacent_position_heatmaps.png", "Peak Adjacent Relative Position Error versus Mechanism Count", "peak_adjacent_relative_position_error_m", "Peak adjacent position error (mm)",1e3,True)),
        ("Figure 6.8", lambda: count_heatmaps(rows, output, "figure_6_8_adjacent_rotation_heatmaps.png", "Peak Adjacent Relative Rotation Error versus Mechanism Count", "peak_adjacent_relative_rotation_error_rad", "Peak adjacent rotation error (mrad)",1e3,True)),
        ("Figure 6.9", lambda: figure_6_9(rows, args.main_sweep_dir, output)),
        ("Figures 6.11–6.12", lambda: figure_6_11_12(srows, output)),
        ("Figures 6.13–6.16", lambda: vertical_figures(args.vertical_dir, output)),
        ("Figures 7.1–7.2", lambda: figure_7_1_2(srows, args.supplemental_dir, output)),
    ]
    for label, function in requests:
        result = function()
        if result is None or result == []:
            skipped.append(label)
        elif isinstance(result, list):
            made.extend(result)
        else:
            made.append(result)
    manifest = {
        "generated": [path.name for path in made],
        "skipped_incomplete": skipped,
        "figure_directory": str(output),
        "plot_title_policy": "No software version identifiers or interpolation-method names",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return made, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-sweep-dir", type=Path, default=MAIN_SWEEP_DIR)
    parser.add_argument("--supplemental-dir", type=Path, default=SUPPLEMENTAL_DIR)
    parser.add_argument("--vertical-dir", type=Path, default=VERTICAL_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    made, skipped = generate(parse_args())
    print(f"Generated {len(made)} figure files.")
    if skipped:
        print("Skipped because required simulations are incomplete:")
        for item in skipped: print(f"  {item}")
