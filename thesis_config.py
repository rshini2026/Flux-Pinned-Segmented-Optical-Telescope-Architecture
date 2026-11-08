"""Authoritative settings for the final thesis figure suite."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "Final_Thesis_Results"
MAIN_SWEEP_DIR = OUTPUT_ROOT / "main_sweep"
SUPPLEMENTAL_DIR = OUTPUT_ROOT / "supplemental"
VERTICAL_DIR = OUTPUT_ROOT / "vertical_placement"
FIGURE_DIR = OUTPUT_ROOT / "figures"

MODELS = ("chiang", "jiang")
TEMPERATURES_K = (50.0, 77.0, 85.0)
FIELD_COOLING_SEPARATIONS_MM = (7.5, 10.0, 25.0)
MECHANISM_COUNTS = (2, 4, 6, 8, 10)

CHAPTER5_COUNT = 4
BASELINE_ROW_HALF_HEIGHT_MM = 22.5
BASELINE_ROW_SEPARATION_MM = 45.0
BASELINE_ALONG_EDGE_HALF_SPAN_MM = 225.0
MECHANISM_DAMPING_RATIO = 0.03
CENTER_RESTRAINT_DAMPING_RATIO = 0.01

IMPULSE_TIME_S = 1.5
DEFAULT_TIMESTEP_S = 2.0e-4
DEFAULT_TRANSIENT_TIME_S = 16.0
DEFAULT_LONG_TIME_S = 120.0


def ensure_output_directories() -> None:
    for path in (OUTPUT_ROOT, MAIN_SWEEP_DIR, SUPPLEMENTAL_DIR,
                 VERTICAL_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)

