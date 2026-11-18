# Validation against standards

This folder checks the `earthing` package against the worked examples, tables and formulas in the reference standards:

| Short name | Document |
| --- | --- |
| IEEE 80 | IEEE Std 80-2000, *IEEE Guide for Safety in AC Substation Grounding* |
| ENA S34 | ENA EREC S34 Issue 2 (2018), *A guide for assessing the rise of earth potential at electrical installations* |
| ENA TS 41-24 | ENA TS 41-24 Issue 2 (2018), *Guidelines for the design, installation, testing and maintenance of main earthing systems in substations* |
| IEEE 142 | IEEE Std 142-2007, *IEEE Recommended Practice for Grounding of Industrial and Commercial Power Systems* (Table 4-5 only) |

Reference values were transcribed from these documents. Each case records its clause, equation and printed page number. The PDFs are not needed to run the checks.

## Running

From the repository root, with `numpy`, `matplotlib` and (for the pytest route) `pytest` installed. Use `python3`; on many Linux systems `python` is not on the PATH.

```bash
python3 validation/run_validation.py
```

```bash
python3 validation/run_validation.py --skip-numerical --report validation/report.md
```

```bash
python3 -m pytest validation
```

```bash
python3 -m pytest validation -m "not numerical"
```

```bash
python3 validation/discretisation_sweep.py
```

`run_validation.py` prints a table grouped by standard, lists the failing cases with their source clause, and exits with status 1 if any case fails. `--only TEXT` runs only cases whose id contains `TEXT`. `--report PATH` also writes a Markdown report. The pytest route runs the same cases, apart from the informational ones.

Use `--skip-numerical` or `-m "not numerical"` to leave out the `Network` solver cases. They take about a minute and need roughly 1 GB of RAM.

## Files

| File | Purpose |
| --- | --- |
| `cases.py` | All validation cases: inputs, reference values, sources and tolerances. Add new cases here. |
| `run_validation.py` | Command-line runner and report writer. |
| `test_standards.py` | pytest wrapper around the same cases. |
| `report.md` | Last generated report (regenerate with `--report`). |
| `discretisation_sweep.py` | Regenerates the tables in Discretisation guidance (under a minute). |

## Coverage

| Function | Checked against |
| --- | --- |
| `e_step_70`, `e_touch_70` | IEEE 80 Eqs. (27), (30), (33); Annex B.1 Step 3 |
| `e_step_50`, `e_touch_50` | IEEE 80 Eqs. (29), (32); Clause 17 fence case study; formula identities |
| `resistance_grid` | IEEE 80 Eq. (52); Annex B Examples 1–4 |
| `e_mesh_step_grid`, `e_mesh_step` | IEEE 80 Eqs. (80)–(94); Annex B Examples 1–4 (Em, Es), including the L-shaped Example 4; Eqs. (82) and (90) for interior rods (formula identity); ENA S34 P4 (case study 6.1.6, informational) |
| `fault_current` | IEEE 80 Annex B.1 Step 2 (3I0 at 115 kV and 13 kV) |
| `resistance_pipe` | ENA S34 Formula R1 (case studies 6.1.1, 6.5.1); IEEE 80 Eq. (55) with n_R = 1 |
| `resistance_plate` | ENA S34 Formula R2 as printed, plus its exact surface and deep-burial limits |
| `resistance_strip` | `method='S34'`: ENA S34 Formula R7; `method='IEEE142'`: IEEE 142 Table 4-5 (formula identities). The IS 3043 default is informational only. |
| `resistance_grid_with_rods` | ENA S34 Formulae R4–R6 (case studies 6.1.1 with and without re-bar, 6.5.1), with `w = π·d` and with `b` passed directly; factor k against Figure B.1 for 4–72 rods |
| `resistance_parallel` | ENA S34 Formula R10 / case study 6.5.6 |
| `current_ratio` | ENA S34 Appendix D.4.1 (case studies 6.2.3 and 6.5.2) |
| `max_current_density` | ENA S34 Formula C2 (case study 6.5.5), both methods; ENA TS 41-24 Table 8 (96 electrode ratings) |
| `e_touch_external`, `e_touch_fence` | ENA S34 Formulae P1, P2 (case studies 6.1.4, 6.1.5) |
| `e_touch_fence_peripheral` | ENA S34 Formula P3 as printed (formula identity only) |
| `e_step_external` | ENA S34 Formula P5, consistency with P6.3 |
| `surface_potential_*` | ENA S34 Formulae P6.1 and P6.3 (identities); P6.2 against Appendix I.5 (259 V, 48 V) |
| `contour_distance` | ENA S34 Formula P7 (case study 6.1.7) |
| `touch_limit_ts4124`, `step_limit_ts4124` | ENA TS 41-24 Tables 1 and 2, including the values quoted in S34 (944 V, 420 V, 233 V); both interpolation modes |
| `Network` solver | ENA S34 R1 (rod) and R2 (plate); EPRI TR-100622 results quoted in IEEE 80 Annex B for Examples 1, 2 and 4 (Rg, Em, Es) |
| `Network.generate_model` | ENA S34 R1, and equality with `generate_model_fast` |
| `Network.get_point_potential` | ENA S34 P6.2 at 1 m, 9 m and 50 m from a rod |
| `Network.add_subnet` | IEEE 142 Table 4-5 two-rod formula (mutual resistance); ENA S34 R1 (energised rod) |

Not covered:

- `resistance_strip` with the IS 3043 default: IS 3043 is not in the reference set.
- `e_touch_fence_peripheral` (S34 P3) and `e_step_external` (S34 P5): S34 gives no worked examples, so only the formulas as printed are checked.
- `e_mesh_step` with n_d ≠ 1 (grids that are not square, rectangular or L-shaped): IEEE 80 gives no worked example.
- `Network` step voltage for IEEE 80 Example 1: IEEE 80 quotes no EPRI step voltage for that example. Example 3 is not modelled.

## Tolerance policy

- **Closed-form functions: ±1 %**, covering rounding in the published values; intermediate factors such as Cs and Km are rounded to two decimals in IEEE 80. Formula-identity checks use 1e-9.
- **Factor k (Figure B.1): ±0.05** where the standard states the value, ±0.15 where it was read from the chart.
- **Numerical solver: ±5 % for resistance, ±10 % for touch and step voltage.** Grid and plate cases use the 0.2 m discretisation from the project notebooks. The single-rod case uses 0.1 m (see Discretisation guidance).
- **Informational cases** (`INFO`) are reported with their deviation but never pass or fail. They show known inconsistencies or sensitivities.

## Findings

Latest results: see `report.md`. See `report.md`. The first recorded run (commit `d773f36`) had 152 cases: 139 passed and 3 failed. The fix for finding 1 turned two failures into passes and added one case.

1. **Fixed: the factor k in `resistance_grid_with_rods` diverged above 64 rods.** The 5th-order polynomial fitted to Figure B.1 gave k = 9.59 at 68 rods, 10.73 at 72 and 72.9 at 100, where the chart reads 9.07, 9.16 and ends at 72 rods. The package now uses monotone piecewise cubic interpolation of the digitised points, with a point added at 72 rods (9.2). For more than 72 rods it holds k at 9.2 and issues a warning. Case `S34-FigB1-k-N100-beyond-chart` guards this.
2. **The `Network` solver depends on discretisation size and does not converge.** A single 3.6 m rod comes out 9 % above Formula R1 at Δ = 0.2 m and 24 % above at 0.4 m. The error is smallest near Δ = 0.05 m and grows again below that: +2.5 % at 0.025 m. Each segment is modelled as a disc of equal area, so the effective conductor radius changes with Δ. Grid results at Δ = 0.2 m agree with EPRI to within 0.2 % for resistance and 6 % for mesh voltage. The rod case `NUM-rod-vs-S34-R1` now runs at Δ = 0.1 m (+1.9 %). At 0.2 m it failed the ±5 % tolerance. See Discretisation guidance below.
3. **ENA S34 is inconsistent about the parameter b in Formula R6.** The document gives three definitions:
   - Appendix B.1.7 (Formula R6): b = w/π, where w is the tape width. The package uses this by default.
   - Appendix A (symbols): b is the "equivalent diameter of circular electrode".
   - Worked examples 6.1.1 and 6.5.1: b is the "equivalent diameter of the circular earth electrode or the width of a tape electrode", and b = 0.01 m (the conductor diameter) is substituted.

   Appendix A and the worked examples agree on an equivalent diameter. Formula P1 defines equivalent diameter as conductor perimeter/π. If w in B.1.7 is read as the conductor perimeter rather than the tape width, b = w/π is the same equivalent diameter, so B.1.7 most likely mislabels w. The package now takes an optional `b` argument. Pass `b = d` for a round conductor of diameter d, or `b = 2(w + t)/π` for a tape of width w and thickness t. With `b = 0.01` the package reproduces all three worked examples (1.62 Ω, 1.43 Ω, 6.57 Ω). The default (b = w/π, with w = tape width) is unchanged for compatibility. Passing `w = d` without `b` gives R_E 5.7 % lower (1.53 Ω instead of 1.62 Ω).
4. **Mesh voltage depends on how a round conductor is turned into a strip.** With strip width 2d (equivalent radius w/4) the IEEE Example 1 mesh voltage is +5.9 % against EPRI. With the equal-area width π·d/2 it is +12.96 %.
5. **`max_current_density` uses the IS 3043 coefficient 7.57e3 by default.** ENA S34 Formula C2 implies 1e3·√57.7 = 7.596e3, so the default limit is 0.34 % lower (conservative). All TS 41-24 Table 8 ratings still agree to within 1 %. `method='S34'` selects Formula C2.
6. **Fixed: `e_mesh_step_grid` always used IEEE 80 Eq. (91) for L_M, with K_ii = 1, whenever rods were present.** That form assumes rods at the corners and along the perimeter. `rods_on_perimeter=False` now selects K_ii from Eq. (82) and L_M from Eq. (90) for rods inside the grid only. The default is unchanged.
7. **`Network.generate_model` can raise `ZeroDivisionError` at small discretisation sizes** (seen for a rod at Δ = 0.00625 m). This happens when an evaluation point lies in an element's plane inside its radius. `generate_model_fast` handles the same case through `arctan(inf)`.
8. **ENA S34 Formula P2 divides by L_p as printed, but worked example 6.1.5 divides by L_T.** With L_T = 176 m the package reproduces the published 169 V; with L_p = 136 m the result would be 218 V. The text also says k_f is "applied in place of k_e in the above formulae", that is, in Formula P1, which divides by L_T. `e_touch_fence` and `e_touch_fence_peripheral` (P3, same structure) use L_T.
9. **ENA S34 Formula P3 has no 1/π factor in k_fe, unlike k_e in P1.** As printed, k_fe is about 2 for typical inputs, against k_e ≈ 0.95. A fence with an extra peripheral conductor would then see a higher touch potential than the grid edge, which is implausible. The package implements P3 as printed. Treat `e_touch_fence_peripheral` results with caution until P3 is confirmed against another source.
10. **ENA S34 case study 6.1.6 does not state the spacing D for Formula P4.** With D = 15 m (the value used for P1 in 6.1.4) `e_mesh_step_grid` gives 689 V, +4.9 % on the published 657 V. D = 12.5 m reproduces 657 V. The case is informational.
11. **IEEE 80 reads C_s from Figure 11 in the Clause 17 fence case study (C_s = 0.62).** The package uses Eq. (27), which gives 0.6355, so the 50 kg limits come out about 2 % higher than the published 1995 V and 622 V. The cases allow ±3 %.

## Discretisation guidance

`generate_model_fast` replaces each segment with a flat circular disc of equal area. The error therefore depends on the step Δ relative to the conductor size, not on Δ alone. Refining Δ does not converge to the exact value: below the optimum the error grows again. The sweeps below compare a single electrode with a closed-form reference. Regenerate them with `python3 validation/discretisation_sweep.py`.

### Rods (`add_rod`)

Reference: ENA S34 Formula R1. Rods of 1.2–7.5 m, diameters 12–25 mm. Deviation in %, by Δ as a multiple of rod diameter d:

| Δ / d | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deviation | +5.2 to +7.1 | +1.1 to +1.3 | +0.1 to +0.3 | +0.2 to +0.8 | +1.4 to +3.9 | +3.2 to +8.2 | +7.2 to +18.9 |

- Use **Δ ≈ 3–4 d** for minimum error (within 1 %). For a 16 mm rod that is about 0.05–0.065 m.
- Δ ≤ 6 d keeps the error within about 2 % for rods 2.4 m and longer, and within 4 % for 1.2 m rods.
- The trend does not depend on rod length. Short, thick rods are the most sensitive to coarse Δ.

### Strips (`add_strip`)

Reference: Dwight formula for a buried horizontal wire (IEEE Std 142-2007 Table 4-5), with equivalent radius a = w/4 and burial depth 0.5 m. Strips of 10–50 m, widths 25–50 mm. Deviation in %, by Δ as a multiple of strip width w:

| Δ / w | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deviation | −1.0 to −1.4 | −2.3 to −3.0 | −2.0 to −2.7 | −1.4 to −1.8 | +0.4 to +0.8 | +2.4 to +3.4 | +10.1 to +13.9 |

- Use **Δ ≈ 4–8 w** for an isolated strip; the error stays within 3 %. For a 25 mm tape that is about 0.1–0.2 m.
- The a = w/4 equivalence itself carries a few per cent of uncertainty, so do not read the sign of small deviations as meaningful.
- Large meshes are less sensitive: the IEEE 80 Annex B grids use Δ = 0.2 m with w = 0.02 m (Δ = 10 w), and resistance still agrees with EPRI to within 0.2 %. Mutual coupling between conductors dominates the result.

### Plates (`add_plate`)

Reference: ENA S34 Formula R2. Horizontal square plates of 0.6–1.2 m side at 0.5–1.0 m depth:

| Δ (m) | 0.4 | 0.2 | 0.1 | 0.05 | 0.025 |
| --- | --- | --- | --- | --- | --- |
| Deviation | +0.4 to +1.5 | −1.7 to +0.3 | −2.6 to −1.7 | −2.8 to −2.2 | −2.9 to −2.2 |

- Plates converge: below Δ ≈ side / 10 the result changes by less than 1 %.
- The −2 to −3 % residual is within the accuracy of R2, which is an approximation itself.
- Use **Δ ≤ side / 5** (0.2 m for a 1 m plate).

### Mixed networks

- One Δ applies to the whole network. Pick it for the most sensitive electrode, usually the rods (Δ ≈ 3–6 d).
- Element count, and so memory, grows as (total conductor length / Δ)². The IEEE 80 70 m × 70 m grid needs about 1 GB at Δ = 0.2 m. Halving Δ needs about four times that.
- When memory limits Δ, accept the rod error in the tables above, or check the result at two Δ values.

