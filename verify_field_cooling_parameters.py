from __future__ import annotations

from pathlib import Path
import numpy as np

from mechanism_parameters import mechanism_C6, mechanism_K6, temperature_ratio


def main() -> None:
    source = (Path(__file__).resolve().parent / "segment_dynamics.py").read_text(
        encoding="utf-8"
    )
    assert "case_K_local = np.asarray(" in source
    assert "mechanism_K6(\n                GEOM.gap" in source
    assert "case_C_local = np.asarray(" in source
    assert "mechanism_C6(\n                GEOM.gap" in source
    assert "K_local=case_K_local" in source
    assert "C_local=case_C_local" in source
    assert source.count("mechanism_K6(") == 2, (
        "Expected one per-case constitutive lookup plus the reporting lookup in main()."
    )
    assert source.count("mechanism_C6(") == 2, (
        "Expected one per-case constitutive lookup plus the reporting lookup in main()."
    )
    assert "mechanism_K6(\n        instantaneous_separation" not in source
    assert "mechanism_C6(\n        instantaneous_separation" not in source

    print("Field-cooling constitutive-matrix check: PASS")
    print("K and C are evaluated from GEOM.gap, not instantaneous separation.\n")
    for model in ("chiang", "jiang"):
        print(f"{model.title()} temperature ratios:")
        for temperature in (50.0, 77.0, 85.0):
            ratio = temperature_ratio(temperature, model)
            K = mechanism_K6(0.010, temperature, model)
            C = mechanism_C6(0.010, temperature, model)
            assert np.all(np.isfinite(K)) and np.all(np.isfinite(C))
            print(
                f"  {temperature:4.0f} K: cT={ratio:.6f}, "
                f"Kpar={K[0,0]:.6f} N/m, Kperp={K[1,1]:.6f} N/m"
            )


if __name__ == "__main__":
    main()

