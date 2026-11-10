# Flux-pinned seven-segment mirror structural model

This repository contains the code used to model a rigid seven-segment mirror
assembly connected by flux-pinning mechanisms. It includes the temperature and
field-cooling-separation parameterization, 42-coordinate modal assembly,
MuJoCo impulse simulation, design sweeps, support comparison, and figure
generation described in the supporting thesis.

The mechanism matrices are evaluated at the selected field-cooling separation
and temperature once per transient. Chiang and Jiang levitation-force
datasets are independently interpolated with PCHIP and normalized at 77 K.
Their force ratios are assumed to scale the translational stiffness. The
primary local mechanism matrix has no direct rotational coefficients;
distributed translational forces generate assembly-level rotational resistance.

## Contents

| File | Role |
| --- | --- |
| `empirical_mechanism_data.py` | Reconstructed stiffness data, separation fits, and temperature inputs |
| `mechanism_parameters.py` | Segment properties and local stiffness and damping matrices |
| `segment_dynamics.py` | Rigid-body MuJoCo geometry and transient force calculation |
| `vertical_placement_study.py` | Global mass, stiffness, and damping assembly; vertical-row study |
| `main_parameter_sweep.py` | Principal 90-case modal and transient sweep |
| `supplemental_studies.py` | Reference, damping, extrapolation, placement, and 16-second alternate-matrix studies |
| `generate_thesis_figures.py` | Numbered figure generation from saved sweep results |
| `run_thesis_pipeline.py` | Resumable orchestration of the preceding thesis studies |
| `alternate_matrix_settling_study.py` | Long-duration 77 K, 25 mm, N=4 alternate-matrix settling comparison |
| `support_boundary_comparison.py` | Exact 3-2-1 modal and penalty-supported transient comparison |
| `support_settling_study.py` | Longer support-boundary settling comparison |
| `reference_timestep_study.py` | Matched MuJoCo timestep comparison for the reference case |
| `verify_field_cooling_parameters.py` | Field-cooling-fixed constitutive lookup check |
| `thesis_config.py` | Common case and output paths |

All Python scripts are deliberately kept in one directory because they import
each other by module filename.

## Installation

Use a Python environment with the four packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the thesis analyses

Run commands from the repository directory. The complete sweep can take several
hours; each completed case is checkpointed and the command can be resumed.

```bash
python verify_field_cooling_parameters.py
python run_thesis_pipeline.py --stage all
```

Results appear under `Final_Thesis_Results/`, with separate `main_sweep/`,
`supplemental/`, `vertical_placement/`, and `figures/` directories. To rebuild
numbered figures from completed cases without recomputing transients:

```bash
python run_thesis_pipeline.py --stage plots
```

```bash
python alternate_matrix_settling_study.py --configuration 77K_25mm_N4
```

The support-boundary study and reference-case timestep check are separate:

```bash
python support_boundary_comparison.py
python support_settling_study.py
python reference_timestep_study.py
```

The timestep script defaults to a four-second interval containing the saved
reference-case peaks. Use `--duration 16` for the full comparison window.

## Scope

The geometry, impulse, assumed damping, and source-derived stiffness inputs
are documented in the thesis. The 40 K, 1 mm case extrapolates the separation
fit below its 5 mm input boundary. The alternate matrix comes from a different
experimental interface and is used as a model-form comparison. Neither
relationship is a validation of the proposed telescope hardware.

Using your own experimental data

The default inputs are literature-derived estimates. Mechanism stiffness is calculated from a 77 K separation curve and a temperature multiplier:

stiffness(separation, temperature) = stiffness_at_77K(separation) × temperature_multiplier(temperature)

The multiplier is normalized to 1 at 77 K. The resulting stiffness is used to construct the local mechanism matrix; mechanism damping is then calculated from that stiffness and the selected damping ratio. Change the inputs below before running a new study.

Mechanism stiffness measured at 77 K

Edit SEP_D_MM, SEP_K11, and SEP_K33 in empirical_mechanism_data.py if you have stiffness measurements at several field-cooling separations. SEP_D_MM is in millimeters; the stiffness arrays are in N/m. Each stiffness entry must correspond to the separation at the same array index. The code fits an exponential separation curve to these points through fit_separation_models(). Provide at least three positive measurements for each fitted direction.

K11 supplies the two lateral entries of the primary local stiffness matrix, and K33 supplies its axial entry. SEP_K22 is also fitted and used by some empirical-data helpers, but the primary mechanism matrix in mechanism_parameters.py uses K11 and K33. Check that your measured directions match the model’s local axes before substituting data.

If you have measurements at only one separation, they cannot establish a new separation curve. To retain the existing curve’s shape while matching your measured 77 K stiffness at a reference separation, rescale K_parallel() and K_perp() in mechanism_parameters.py. For example, for measurements at 10 mm, return the measured stiffness multiplied by fit.stiffness(d_m) / fit.stiffness(0.010). This matches the measurement at 10 mm but retains the original assumed change with separation.

Temperature dependence

The runtime multiplier is selected by temperature_ratio() in mechanism_parameters.py. Its existing Chiang and Jiang branches interpolate the temperature and levitation-force arrays in empirical_mechanism_data.py and divide by the interpolated value at 77 K. If your measurements are comparable force measurements, replace the temperature and force arrays for the branch you intend to run. Temperatures are in kelvin, and each force entry must correspond to the temperature at the same index.

If you have measured stiffness versus temperature, use ratios of stiffness to its 77 K value instead of treating the literature force data as your measurements. Update temperature_ratio() to interpolate those ratios, ensure the result is 1 at 77 K, and adjust its temperature-range check to match your measured range. The current function accepts only the model names "chiang" and "jiang"; adding a new name also requires updating the model choices in the study scripts and thesis_config.py. If the lateral and axial stiffnesses have different temperature trends, modify translational_stiffnesses() to apply a separate measured multiplier to each direction.

Update figure labels and data markers if you replace the Chiang or Jiang datasets; otherwise, generated plots may still describe the new inputs as literature data.

Backplane or central restraint

The default model restrains the central segment with translational and rotational springs and dampers. In segment_dynamics.py, DynamicsConfig.anchor_k_trans and anchor_k_rot set their stiffnesses in N/m and N·m/rad. anchor_zeta_trans and anchor_zeta_rot set damping ratios. anchor_mass and anchor_inertia are used to convert those ratios into damping coefficients in anchor_parameters(). apply_anchor() applies the resulting forces and torques during a transient. The modal assembly reads the same restraint settings in vertical_placement_study.py, in build_linear_system().

Replace these settings with values identified for your backplane if a central spring and damper is an appropriate representation. A backplane connected to multiple segments, or one with directional or coupled behavior, requires changes to both the transient forces in segment_dynamics.py and the assembled stiffness and damping matrices in vertical_placement_study.py. Changing only one would make the modal and transient analyses represent different structures.

The separate support_boundary_comparison.py study overrides the central restraint with its own model_settings() values of 500 N/m, 500 N·m/rad, and 1% damping. Update that function as well if you want its center-restraint case to represent your backplane. Its supports() function and support options describe the separate 3-2-1 comparison, not the central restraint used by the main sweep.

Separation measurements and cases

To replace the stiffness-versus-separation relationship, edit SEP_D_MM, SEP_K11, and SEP_K33 as described above. The fitted curves are created when mechanism_parameters.py is imported, so restart Python after changing the arrays.

To change which separations are simulated, edit FIELD_COOLING_SEPARATIONS_MM in thesis_config.py and the explicit --separations-mm values passed by run_thesis_pipeline.py. Individual study scripts may also define their own cases or command-line defaults. The input separation describes the field-cooling condition used to select a fixed mechanism stiffness and damping matrix for each transient; the instantaneous gap is recorded as a motion diagnostic and does not update that matrix.

The existing separation-domain helper in mechanism_parameters.py contains a fixed upper limit of 30 mm, and several study cases and plots retain the original separation choices. Review those limits, cases, and labels when extending the measured range. Treat predictions outside the range of your measurements as extrapolations.

Recomputing results

Run python verify_field_cooling_parameters.py after changing the inputs, then run the desired studies. Saved cases are resumable and may be skipped on a normal rerun. Use python run_thesis_pipeline.py --stage all --force to recompute the main pipeline after a parameter change, preferably with a separate output directory or an archived copy of the earlier results. Rerun separately launched studies, including the alternate-matrix and support comparisons, when their inputs change. Record the measurements, units, fit method, and code revision alongside any results you publish.

