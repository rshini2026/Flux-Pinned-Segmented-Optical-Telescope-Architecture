from __future__ import annotations

"""Run 80-second 3-2-1 versus 1% center-restraint settling histories.

Fixed physical configuration:
    temperature = 50 K
    field-cooling separation = 7.5 mm
    mechanisms per shared interface = 10
    vertical rows = +/-22.5 mm
    along-edge half-span = 225 mm

Both Chiang-PCHIP and Jiang-PCHIP temperature sensitivities are evaluated.
The center restraint has 1% damping. The 3-2-1 modal constraint is exact;
its MuJoCo transient uses the finite penalty support with zero damping by
default. The mechanism damping ratio is 0.03 in every case.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import support_boundary_comparison as comparison


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "Support_Settling_50K_d7p5mm_N10_Output"
MODELS = (("chiang", "Chiang PCHIP"), ("jiang", "Jiang PCHIP"))
BOUNDARIES = (
    ("center_1pct", "1% center restraint"),
    ("kinematic_321", "3-2-1 support (zero support damping)"),
)
TEMPERATURE_K = 50.0
SEPARATION_MM = 7.5
MECHANISM_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-time", type=float, default=80.0)
    parser.add_argument("--timestep", type=float, default=1.0e-4)
    parser.add_argument("--support-stiffness", type=float, default=1.0e4)
    parser.add_argument("--kinematic-support-zeta", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def case_key(model: str, boundary: str) -> str:
    return f"{model}|{boundary}"


def observed_settling_metrics(t: np.ndarray, y: np.ndarray,
                              impulse_time: float = 1.5) -> dict:
    """Return a conservative recorded-window 2% settling diagnostic.

    The response metrics are nonnegative magnitudes. Their reverse cumulative
    maximum is used as an upper envelope. A case is marked settled only if that
    envelope falls below 2% of the post-impulse peak and remains there through
    the end of the recorded interval.
    """
    mask = t >= impulse_time
    tp = np.asarray(t[mask], dtype=float)
    yp = np.asarray(y[mask], dtype=float)
    if len(tp) == 0 or not np.all(np.isfinite(yp)):
        raise RuntimeError("Invalid post-impulse history for settling calculation.")
    peak = float(np.max(yp))
    threshold = 0.02 * peak
    envelope = np.maximum.accumulate(yp[::-1])[::-1]
    valid = np.flatnonzero(envelope <= threshold)
    settled = bool(len(valid))
    settling_absolute = float(tp[valid[0]]) if settled else None
    return {
        "peak": peak,
        "threshold": threshold,
        "settled_within_record": settled,
        "settling_absolute_time_s": settling_absolute,
        "settling_time_after_impulse_s": (
            None if settling_absolute is None else settling_absolute - impulse_time
        ),
        "final_value": float(yp[-1]),
        "final_fraction_of_peak": float(yp[-1] / peak) if peak > 0 else 0.0,
        "envelope_final_fraction_of_peak": (
            float(envelope[-1] / peak) if peak > 0 else 0.0
        ),
        "envelope": envelope,
        "post_time": tp,
    }


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: (row["model"], row["boundary"]))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate_history(output: Path, transient: dict) -> dict:
    with np.load(output / transient["history_file"]) as history:
        t = np.asarray(history["t"], dtype=float)
        if len(t) == 0:
            raise RuntimeError("The transient returned an empty history.")
        metrics = {}
        for key in (
            "mean_position_m", "mean_rotation_rad",
            "adjacent_position_m", "adjacent_rotation_rad",
        ):
            result = observed_settling_metrics(t, np.asarray(history[key]))
            prefix = key.removesuffix("_m").removesuffix("_rad")
            metrics[f"{prefix}_settled_within_80s_record"] = result["settled_within_record"]
            metrics[f"{prefix}_2pct_settling_after_impulse_s"] = result["settling_time_after_impulse_s"]
            metrics[f"{prefix}_final_fraction_of_peak"] = result["final_fraction_of_peak"]
        metrics["history_final_time_s"] = float(t[-1])
        return metrics


def boundary_label(boundary: str) -> str:
    return dict(BOUNDARIES)[boundary]


def plot_model_histories(output: Path, model: str, model_label: str,
                         rows: list[dict]) -> None:
    selected = [row for row in rows if row["model"] == model]
    if len(selected) != 2:
        return
    panels = (
        ("mean_position_m", 1e3, "Mean position error (mm)"),
        ("mean_rotation_rad", 1e3, "Mean rotation error (mrad)"),
        ("adjacent_position_m", 1e3, "Maximum adjacent position error (mm)"),
        ("adjacent_rotation_rad", 1e3, "Maximum adjacent rotation error (mrad)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                             constrained_layout=True)
    for row in selected:
        with np.load(output / row["history_file"]) as h:
            for ax, (key, scale, ylabel) in zip(axes.flat, panels):
                ax.plot(h["t"], scale*h[key], linewidth=1.2,
                        label=boundary_label(row["boundary"]))
                ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
                ax.axvline(1.5, color="0.4", linestyle=":", linewidth=1)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    axes[0,0].legend()
    fig.suptitle(
        f"80-s Settling Histories — {model_label}\n"
        "50 K, field-cooling separation=7.5 mm, N=10"
    )
    fig.savefig(output / f"figure_long_settling_{model}_histories.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_envelopes(output: Path, rows: list[dict]) -> None:
    panels = (
        ("mean_position_m", "Mean position"),
        ("mean_rotation_rad", "Mean rotation"),
        ("adjacent_position_m", "Adjacent position"),
        ("adjacent_rotation_rad", "Adjacent rotation"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = {"chiang": "C0", "jiang": "C1"}
    styles = {"center_1pct": "-", "kinematic_321": "--"}
    for row in rows:
        with np.load(output / row["history_file"]) as h:
            t = np.asarray(h["t"])
            for ax, (key, title) in zip(axes.flat, panels):
                result = observed_settling_metrics(t, np.asarray(h[key]))
                peak = result["peak"]
                normalized = result["envelope"] / peak if peak > 0 else result["envelope"]
                ax.semilogy(
                    result["post_time"]-1.5, np.maximum(normalized, 1e-6),
                    color=colors[row["model"]], linestyle=styles[row["boundary"]],
                    linewidth=1.6,
                    label=f"{row['model_label']} — {boundary_label(row['boundary'])}",
                )
                ax.set_title(title); ax.grid(alpha=0.25, which="both")
                ax.axhline(0.02, color="0.25", linestyle=":", linewidth=1.2)
    for ax in axes[1]: ax.set_xlabel("Time after impulse (s)")
    for ax in axes[:,0]: ax.set_ylabel("Remaining envelope / peak")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=8)
    fig.suptitle("Normalized 80-s Response Envelopes and 2% Threshold")
    fig.savefig(output / "figure_long_settling_normalized_envelopes.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_final_fractions(output: Path, rows: list[dict]) -> None:
    fields = (
        ("mean_position_final_fraction_of_peak", "Mean position"),
        ("mean_rotation_final_fraction_of_peak", "Mean rotation"),
        ("adjacent_position_final_fraction_of_peak", "Adjacent position"),
        ("adjacent_rotation_final_fraction_of_peak", "Adjacent rotation"),
    )
    labels = [f"{r['model_label']}\n{boundary_label(r['boundary'])}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, fields):
        values = [100*float(row[field]) for row in rows]
        bars = ax.bar(x, values, color=["C0" if r["model"] == "chiang" else "C1" for r in rows])
        ax.axhline(2.0, color="0.25", linestyle=":", label="2% threshold")
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("Final value (% of peak)")
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    axes[0,0].legend()
    fig.suptitle("Response Remaining at the End of the 80-s Record")
    fig.savefig(output / "figure_long_settling_final_fraction.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_plots(output: Path, rows: list[dict]) -> None:
    for model, label in MODELS:
        plot_model_histories(output, model, label, rows)
    plot_normalized_envelopes(output, rows)
    plot_final_fractions(output, rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "long_settling_results.json"
    rows = load_rows(results_path)
    done = {case_key(row["model"], row["boundary"]) for row in rows}
    processed = 0

    if not args.plots_only:
        for model, model_label in MODELS:
            modal = comparison.modal_results(
                model, TEMPERATURE_K, SEPARATION_MM, MECHANISM_COUNT,
                args.support_stiffness, args.kinematic_support_zeta,
            )
            for boundary, _ in BOUNDARIES:
                key = case_key(model, boundary)
                if key in done and not args.force:
                    print(f"SKIP {key}", flush=True)
                    continue
                if args.max_cases is not None and processed >= args.max_cases:
                    save_rows(results_path, rows)
                    print("Partial study saved. Re-run the same command to resume.")
                    return
                print(
                    f"RUN {model_label} | {boundary_label(boundary)} | "
                    f"{args.run_time:g} s",
                    flush=True,
                )
                transient = comparison.run_transient(
                    args, f"{model}_T50K_d7p5mm_N10", model,
                    TEMPERATURE_K, SEPARATION_MM, MECHANISM_COUNT, boundary,
                )
                diagnostics = evaluate_history(args.output_dir, transient)
                if diagnostics["history_final_time_s"] < 0.995*args.run_time:
                    raise RuntimeError(
                        f"{key} ended at {diagnostics['history_final_time_s']:.3f} s "
                        f"instead of {args.run_time:.3f} s; the case was not accepted."
                    )
                row = {
                    "model": model, "model_label": model_label,
                    "boundary": boundary,
                    "temperature_K": TEMPERATURE_K,
                    "field_cooling_separation_mm": SEPARATION_MM,
                    "mechanism_count_per_interface": MECHANISM_COUNT,
                    "run_time_s": args.run_time, "timestep_s": args.timestep,
                    "field_cooling_matrices_fixed_during_transient": True,
                    "mechanism_damping_ratio": comparison.MECHANISM_ZETA,
                    "center_restraint_damping_ratio": comparison.CENTER_RESTRAINT_ZETA,
                    "kinematic_penalty_support_damping_ratio": args.kinematic_support_zeta,
                    "center_frequency_hz": modal["center"]["frequency_hz"],
                    "kinematic_exact_frequency_hz": modal["kinematic_exact"]["frequency_hz"],
                    "kinematic_penalty_frequency_hz": modal["kinematic_penalty"]["frequency_hz"],
                    **transient, **diagnostics,
                }
                rows = [r for r in rows if case_key(r["model"], r["boundary"]) != key] + [row]
                save_rows(results_path, rows)
                done.add(key); processed += 1

    if len(rows) == len(MODELS)*len(BOUNDARIES):
        rows = sorted(rows, key=lambda row: (row["model"], row["boundary"]))
        make_plots(args.output_dir, rows)
        print(f"Complete. Outputs: {args.output_dir}")
    else:
        print("Partial study saved. Re-run the same command to resume.")


if __name__ == "__main__":
    main()
