"""Run the complete final thesis simulation and figure-production pipeline.

Every simulation runner writes its checkpoint after each completed case. The
same command can therefore be stopped with Ctrl+C and run again to resume.
The figure pass is also run after every stage and on an orderly interruption,
so all figures supported by the cases completed so far remain available.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from thesis_config import (
    DEFAULT_LONG_TIME_S,
    DEFAULT_TIMESTEP_S,
    MAIN_SWEEP_DIR,
    SUPPLEMENTAL_DIR,
    VERTICAL_DIR,
    ensure_output_directories,
)


ROOT = Path(__file__).resolve().parent


def command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(ROOT / script), *(str(value) for value in arguments)]


def run_checked(label: str, args: list[str]) -> None:
    print("\n" + "=" * 96, flush=True)
    print(label, flush=True)
    print("=" * 96, flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def refresh_figures() -> None:
    run_checked("COLLECTING ALL CURRENTLY AVAILABLE THESIS FIGURES", command(
        "generate_thesis_figures.py",
        "--main-sweep-dir", MAIN_SWEEP_DIR,
        "--supplemental-dir", SUPPLEMENTAL_DIR,
        "--vertical-dir", VERTICAL_DIR,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "main", "supplemental", "vertical", "plots"),
        default="all", help="Run the full pipeline or one independently resumable stage.",
    )
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP_S)
    parser.add_argument("--long-runtime", type=float, default=DEFAULT_LONG_TIME_S)
    parser.add_argument("--max-main-cases", type=int,
                        help="Optional testing limit; omitted for the final sweep.")
    parser.add_argument("--max-supplemental-cases", type=int,
                        help="Optional testing limit; omitted for final results.")
    parser.add_argument("--max-vertical-cases", type=int,
                        help="Optional testing limit; omitted for final results.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute completed cases instead of resuming them.")
    return parser.parse_args()


def optional_flag(args: list[str], name: str, value: object | None) -> None:
    if value is not None:
        args.extend([name, str(value)])


def main() -> int:
    args = parse_args()
    ensure_output_directories()
    stages = ("main", "supplemental", "vertical") if args.stage == "all" else (args.stage,)
    try:
        if "plots" in stages:
            refresh_figures()
            return 0

        run_checked(
            "PRE-RUN CHECK: FIELD-COOLING MATRICES REMAIN FIXED DURING TRANSIENTS",
            command("verify_field_cooling_parameters.py"),
        )

        if "main" in stages:
            cmd = command(
                "main_parameter_sweep.py",
                "--phase", "both",
                "--models", "chiang", "jiang",
                "--counts", 2, 4, 6, 8, 10,
                "--temperatures", 50, 77, 85,
                "--separations-mm", 7.5, 10, 25,
                "--half-span-mm", 225,
                "--run-time", 16,
                "--max-run-time", 60,
                "--runtime-mode", "one-period",
                "--timestep", args.timestep,
                "--output-dir", MAIN_SWEEP_DIR,
            )
            optional_flag(cmd, "--max-cases", args.max_main_cases)
            if args.force: cmd.append("--force")
            run_checked("STAGE 1 OF 3: MAIN TEMPERATURE, SEPARATION, AND COUNT SWEEP", cmd)
            refresh_figures()

        if "supplemental" in stages:
            cmd = command(
                "supplemental_studies.py",
                "--studies", "all",
                "--long-runtime", args.long_runtime,
                "--timestep", args.timestep,
                "--output-dir", SUPPLEMENTAL_DIR,
            )
            optional_flag(cmd, "--max-cases", args.max_supplemental_cases)
            if args.force: cmd.append("--force")
            run_checked("STAGE 2 OF 3: CHAPTER 5 AND MODEL-FORM SUPPLEMENTAL STUDIES", cmd)
            refresh_figures()

        if "vertical" in stages:
            cmd = command(
                "vertical_placement_study.py",
                "--phase", "both",
                "--model", "chiang",
                "--temperature", 50,
                "--separation-mm", 7.5,
                "--mechanism-count", 10,
                "--z-mm", 22.5, 20, 17.5,
                "--half-span-mm", 225,
                "--run-time", 16,
                "--timestep", args.timestep,
                "--output-dir", VERTICAL_DIR,
            )
            optional_flag(cmd, "--max-cases", args.max_vertical_cases)
            if args.force: cmd.append("--force")
            run_checked("STAGE 3 OF 3: VERTICAL MECHANISM-ROW PLACEMENT", cmd)
            refresh_figures()

    except KeyboardInterrupt:
        print("\nRun interrupted by user. Completed cases remain checkpointed.", flush=True)
        try:
            refresh_figures()
        except Exception as plot_error:
            print(f"Could not refresh partial figures: {plot_error}", flush=True)
        print("Run this same command again to resume at the next incomplete case.", flush=True)
        return 130
    except subprocess.CalledProcessError as error:
        print(f"\nA stage exited with code {error.returncode}.", flush=True)
        print("Completed cases remain checkpointed. Correct the reported issue and rerun to resume.", flush=True)
        try:
            refresh_figures()
        except Exception:
            pass
        return int(error.returncode or 1)

    print("\nFinal thesis simulation and figure pipeline is complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
