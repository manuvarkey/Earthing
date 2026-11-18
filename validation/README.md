# Validation

This folder checks the `earthing` package against worked examples, tables and formulas from published earthing standards.

**Current status:** 224 cases, 210 pass, 0 fail, 14 informational. See [report.md](report.md) for every case.

## Quick start

From the repository root, with `numpy` and `matplotlib` installed:

```bash
python3 validation/run_validation.py                      # all cases (about a minute, about 1 GB RAM)
python3 validation/run_validation.py --skip-numerical     # formula checks only, under a second
```

The pytest route runs the same cases; it needs `pytest` installed:

```bash
python3 -m pytest validation
python3 -m pytest validation -m "not numerical"
```

Use `python3`: on many Linux systems `python` is not on the PATH.

Other options for `run_validation.py`:

- `--only TEXT` runs only the cases whose id contains `TEXT`, for example `--only IEEE80-B4`.
- `--report PATH` also writes a Markdown report. Regenerate the committed report with `--report validation/report.md`.
- The runner exits with status 1 if any case fails.

## Standards used

| Short name | Document |
| --- | --- |
| IEEE 80 | IEEE Std 80-2000, *IEEE Guide for Safety in AC Substation Grounding* |
| IEEE 142 | IEEE Std 142-2007, *Grounding of Industrial and Commercial Power Systems* (Table 4-5, Eq. 4.2) |
| ENA S34 | ENA EREC S34 Issue 2 (2018), *Assessing the rise of earth potential at electrical installations* |
| ENA TS 41-24 | ENA TS 41-24 Issue 2 (2018), *Main earthing systems in substations* |
| IS 3043 | IS 3043:2018, *Code of Practice for Earthing* |

Reference values were copied from these documents. Each case in `cases.py` records its clause, equation and page. The PDFs are not needed to run the checks.

## Kinds of check

- **Worked example:** the package result is compared with a number printed in a standard.
- **Formula identity:** the package result is compared with the formula written out again independently in `cases.py`. Used where a standard gives a formula but no worked example.
- **Solver benchmark:** the numerical `Network` solver is compared with published computer results or closed-form formulas.
- **Informational:** the deviation is reported, but the case never passes or fails. Used to show differences between standards and sensitivities.

## What is checked

### Formula functions

| Area | Functions | Checked against |
| --- | --- | --- |
| Touch and step limits | `e_step_50`, `e_touch_50`, `e_step_70`, `e_touch_70` | IEEE 80 Annex B.1 and Clause 17; IS 3043 clause 10.3 |
| | `touch_limit_ts4124`, `step_limit_ts4124` | ENA TS 41-24 Tables 1 and 2, including the values quoted in ENA S34 |
| Mesh and step voltage | `e_mesh_step_grid`, `e_mesh_step` | IEEE 80 Annex B Examples 1–4, including the L-shaped Example 4; interior-rod equations (82) and (90) |
| Grid resistance | `resistance_grid` | IEEE 80 Eq. (52), Annex B Examples 1–4 |
| | `resistance_grid_with_rods`, `resistance_grid_schwarz` | IEEE 80 Eqs. (53)–(56) (default); ENA S34 R4–R6 worked examples 6.1.1 and 6.5.1 (`method='S34'`); S34 Figure B.1 factor k for 4–72 rods |
| Electrode resistance | `resistance_pipe` | ENA S34 R1 worked examples; IEEE 80 Eq. (55) |
| | `resistance_plate` | IEEE 142 Table 4-5 (default); ENA S34 R2 and its limits (`method='S34'`) |
| | `resistance_strip` | IEEE 142 Table 4-5 (default); ENA S34 R7 and IS 3043:2018 clause 14.2.3 (`method='S34'`) |
| | `resistance_parallel` | ENA S34 R10, worked example 6.5.6 |
| Surface potentials | `e_touch_external`, `e_touch_fence`, `contour_distance`, `surface_potential_rod` | ENA S34 P1, P2, P7 and P6.2 worked examples |
| | `e_touch_fence_peripheral`, `e_step_external`, `surface_potential_hemisphere`, `surface_potential_grid` | ENA S34 P3, P5, P6.1 and P6.3, formula identities only |
| Current | `max_current_density` | ENA S34 C2 (default) and TS 41-24 Table 8 (96 ratings); IS 3043 clause 15.3; IEEE 142 Eq. (4.2) |
| | `fault_current`, `current_ratio` | IEEE 80 Annex B.1; ENA S34 Appendix D worked examples |

### Numerical solver (`Network`, `generate_model_fast`)

| Case | Reference | Step | Deviation |
| --- | --- | --- | --- |
| IEEE 80 Example 1: grid, no rods | EPRI R = 2.67 Ω, Em = 984.3 V | 0.2 m | R +0.2 %, Em +5.9 % |
| IEEE 80 Example 2: grid, 20 rods | EPRI R = 2.52 Ω, Em = 756.2 V, Es = 459.1 V | 0.2 m | R 0.0 %, Em +3.2 %, Es −3.3 % |
| IEEE 80 Example 3: 63 m × 84 m, 38 rods | EPRI R = 2.28 Ω, Em = 519.4 V, Es = 349.7 V | 0.2 m | R +0.2 %, Em +4.7 %, Es +1.9 % |
| IEEE 80 Example 4: L-shaped, 24 rods | EPRI R = 2.34 Ω, Em = 742.9 V, Es = 441.8 V | 0.2 m | R +0.2 %, Em +4.8 %, Es +0.7 % |
| Single rod, 3.6 m | ENA S34 R1 | 0.1 m | +1.9 % |
| Plate, 1 m × 1 m at 0.5 m | ENA S34 R2 / IEEE 142 | 0.2 m | −0.9 % / −1.8 % |
| Strip, 20 m at 0.5 m | IEEE 142 | 0.2 m | +1.7 % |
| Surface potential 1 m, 9 m, 50 m from a rod | ENA S34 P6.2 | 0.05 m | −2.6 %, −0.2 %, 0.0 % |
| Two sub-networks (`add_subnet`) | IEEE 142 two-rod formula | 0.05 m | −0.1 % |
| `generate_model` (slower path) | ENA S34 R1, and equal to `generate_model_fast` | 0.1 m | +1.9 % |

EPRI values are the EPRI TR-100622 computer results quoted in IEEE 80 Annex B. Rod positions for Examples 3 and 4 were read from Figures B.3 and B.4.

### Not checked

- ENA S34 P3 and P5: S34 has no worked examples, so only the formulas as printed are checked.
- `e_mesh_step` for irregular grids other than L-shapes: IEEE 80 has no worked example.
- Solver step voltage for IEEE 80 Example 1: IEEE 80 gives no EPRI value for it.

## Tolerances

| Check | Tolerance |
| --- | --- |
| Worked examples | ±1 %, for rounding in the published values |
| Formula identities | 1e-9 (relative) |
| Figure B.1 factor k | ±0.05 where S34 states the value, ±0.15 where read from the chart |
| Solver resistance | ±5 % |
| Solver touch and step voltage | ±10 % |

A few cases have a wider tolerance, with the reason given in the case note: the IEEE 80 50 kg limits (±3 %, see issue 6 below) and the Example 4 step voltage (±1.5 %, because IEEE rounds Ks and Ki before substituting).

## Issues found in the standards

These are recorded so that results can be traced. No published amendments addressing them were found in November 2026.

1. **ENA S34 R6 defines `b` three ways.** Appendix B prints b = w/π; Appendix A and the worked examples use the conductor's equivalent diameter. With `method='S34'`, `resistance_grid_with_rods` uses w/π unless `b=` is passed; pass it to reproduce the worked examples (`b=0.01` gives 1.62 Ω, 1.43 Ω and 6.57 Ω).
2. **ENA S34 P2 prints L_p as the divisor, but worked example 6.1.5 uses L_T.** Only L_T reproduces the published 169 V (L_p gives 218 V). The package uses L_T for P2 and P3. Drafts of S34 from 2016 show the divisor as an undefined "L".
3. **ENA S34 P3 has no 1/π factor**, unlike P1. As printed it gives a higher touch voltage at the fence than at the grid edge, which is implausible. The package implements it as printed; use with caution.
4. **ENA S34 worked example 6.1.6 does not state the spacing D for P4.** The package gives 689 V against the published 657 V with D = 15 m; D = 12.5 m reproduces 657 V. Informational case.
5. **IEEE 80 Figure 25(b)** prints k2 = −0.05x + 4.40 for curve C, but the plotted curve rises with x. The package uses the printed equation. Curve C only applies to grids deeper than √A/10, so normal substation grids are not affected.
6. **IEEE 80 Clause 17** reads Cs = 0.62 from Figure 11, while Eq. (27) gives 0.6355. The package uses Eq. (27), so its 50 kg limits are about 2 % above the published 1995 V and 622 V.
7. **IS 3043:2018 clause 14.2.1** defines plate area as both sides, but its 1.2 m plate example (ρ/2.75) only works with one side.
8. **IS 3043:2018 clause 14.2.2** prints ln(2l/d) for a rod, while clause 16.4.1 of the same standard uses ln(4L/d). Probably a typo. The package does not use the IS 3043 rod formula.

## Differences between methods

The package offers more than one formula for some electrodes. Results differ:

| Electrode | Default | Alternative | Difference |
| --- | --- | --- | --- |
| Grid with rods (IEEE 80 Example 2) | Schwarz, 2.84 Ω | ENA S34 R4–R6, 2.71 Ω | Schwarz is 13–19 % above EPRI for Examples 2–4; S34 is 8 % above for Example 2 |
| Strip, 20 m × 25 mm at 0.5 m | IEEE 142, 8.45 Ω | ENA S34 R7, 6.42 Ω | R7 has no depth term; the solver gives 8.59 Ω |
| Plate, 1 m² at 0.5 m | IEEE 142, 29.30 Ω | ENA S34 R2, 29.05 Ω | Within 1 %; the solver gives 28.79 Ω |
| Current density, 16 mm rod, 100 Ω·m, 1 s | ENA S34 C2, 38 A/m | IEEE 142 Eq. 4.2, 182 A/m | IEEE 142 is about 4.8 times higher; IS 3043 is 0.3 % below S34 |

## Choosing the solver step size

`generate_model_fast` models each conductor segment as a flat disc. The error depends on the step size Δ relative to the conductor size, and it does not keep falling as Δ gets smaller: below the best step it grows again.

| Electrode | Best step | Error at best step | Error at other steps |
| --- | --- | --- | --- |
| Rod | Δ ≈ 3–4 × rod diameter (0.05–0.065 m for a 16 mm rod) | under 1 % | +1 to +4 % at 6 d; +5 to +7 % at 1 d; +7 to +19 % at 12 d |
| Strip | Δ ≈ 4–8 × strip width (0.1–0.2 m for a 25 mm tape) | within 3 % | +10 to +14 % at 16 w |
| Plate | Δ ≤ side / 5 (0.2 m for a 1 m plate) | within 2 % | converges below side / 10 |

- One step applies to the whole network. Choose it for the most sensitive electrode, usually the rods.
- Memory grows with the square of the number of elements. The IEEE 80 70 m × 70 m grid needs about 1 GB at Δ = 0.2 m; halving Δ needs about four times that.
- Large grids are less sensitive than single electrodes: the Annex B grids at Δ = 0.2 m (10 × strip width) still match EPRI resistance to within 0.2 %.
- When memory limits the step, check the result at two step sizes.

The full sweep tables (rods of 1.2–7.5 m and 12–25 mm, strips of 10–50 m, plates of 0.6–1.2 m) are printed by:

```bash
python3 validation/discretisation_sweep.py
```

## Package issues found and fixed

- Factor k in `resistance_grid_with_rods` (S34 method) diverged above 64 rods; it is now interpolated from Figure B.1 and held at the chart end value beyond 72 rods.
- `e_mesh_step_grid` always assumed perimeter rods; `rods_on_perimeter=False` now applies IEEE 80 Eqs. (82) and (90).
- The single-rod solver case failed at Δ = 0.2 m (+9 %); it now runs at 0.1 m (+1.9 %).

Still open: `Network.generate_model` can raise `ZeroDivisionError` at very small steps (seen at Δ = 0.00625 m). `generate_model_fast` does not have this problem.

## Files

| File | Purpose |
| --- | --- |
| `cases.py` | All cases: inputs, reference values, sources and tolerances. Add new cases here. |
| `run_validation.py` | Command-line runner and report writer. |
| `test_standards.py`, `conftest.py` | pytest wrapper around the same cases. |
| `report.md` | Latest full report. |
| `discretisation_sweep.py` | Prints the solver step-size tables. |
