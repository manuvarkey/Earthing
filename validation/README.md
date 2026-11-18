# Validation against standards

This folder checks the `earthing` package against the worked examples, tables and formulas in the reference standards:

| Short name | Document |
| --- | --- |
| IEEE 80 | IEEE Std 80-2000, *IEEE Guide for Safety in AC Substation Grounding* |
| ENA S34 | ENA EREC S34 Issue 2 (2018), *A guide for assessing the rise of earth potential at electrical installations* |
| ENA TS 41-24 | ENA TS 41-24 Issue 2 (2018), *Guidelines for the design, installation, testing and maintenance of main earthing systems in substations* |

Reference values were transcribed from these documents. Each case records its clause, equation and printed page number. The PDFs are not needed to run the checks.

## Running

From the repository root, with `numpy`, `matplotlib` and (for the pytest route) `pytest` installed:

```bash
python validation/run_validation.py
```

```bash
python validation/run_validation.py --skip-numerical --report validation/report.md
```

```bash
pytest validation
```

```bash
pytest validation -m "not numerical"
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

## Coverage

| Function | Checked against |
| --- | --- |
| `e_step_70`, `e_touch_70` | IEEE 80 Eqs. (27), (30), (33); Annex B.1 Step 3 |
| `resistance_grid` | IEEE 80 Eq. (52); Annex B Examples 1–4 |
| `e_mesh_step_grid` | IEEE 80 Eqs. (80)–(94); Annex B Examples 1–3 (Em, Es) |
| `fault_current` | IEEE 80 Annex B.1 Step 2 (3I0 at 115 kV and 13 kV) |
| `resistance_pipe` | ENA S34 Formula R1 (case studies 6.1.1, 6.5.1); IEEE 80 Eq. (55) with n_R = 1 |
| `resistance_plate` | ENA S34 Formula R2 as printed, plus its exact surface and deep-burial limits |
| `resistance_grid_with_rods` | ENA S34 Formulae R4–R6 (case studies 6.1.1, 6.5.1); factor k against Figure B.1 for 4–72 rods |
| `resistance_parallel` | ENA S34 Formula R10 / case study 6.5.6 |
| `current_ratio` | ENA S34 Appendix D.4.1 (case studies 6.2.3 and 6.5.2) |
| `max_current_density` | ENA S34 Formula C2 (case study 6.5.5); ENA TS 41-24 Table 8 (96 electrode ratings) |
| `Network` solver | ENA S34 R1 (rod) and R2 (plate); EPRI TR-100622 results quoted in IEEE 80 Annex B for Examples 1 and 2 (Rg, Em, Es) |

Not covered:

- `resistance_strip` cites IS 3043, which is not in the reference set. It is shown for information only, next to ENA S34 Formula R7 (a different formula) and the numerical solver.
- `e_mesh_step_grid` handles rectangular grids only, so IEEE 80 Example 4 (L-shaped) checks `resistance_grid` alone.
- The ENA S34 surface-potential formulae P1–P7 and the ENA TS 41-24 touch/step limits (Tables 1 and 2) are not implemented in the package.

## Tolerance policy

- **Closed-form functions: ±1 %**, covering rounding in the published values; intermediate factors such as Cs and Km are rounded to two decimals in IEEE 80. Formula-identity checks use 1e-9.
- **Factor k (Figure B.1): ±0.05** where the standard states the value, ±0.15 where it was read from the chart.
- **Numerical solver: ±5 % for resistance, ±10 % for touch and step voltage.** Grid and plate cases use the 0.2 m discretisation from the project notebooks. The single-rod case uses 0.1 m (see Discretisation guidance).
- **Informational cases** (`INFO`) are reported with their deviation but never pass or fail. They show known inconsistencies or sensitivities.

## Findings

Latest results: 153 cases, 143 passed, 0 failed, 10 informational. See `report.md`. The first recorded run (commit `d773f36`) had 152 cases: 139 passed and 3 failed. The fix for finding 1 turned two failures into passes and added one case.

1. **Fixed: the factor k in `resistance_grid_with_rods` diverged above 64 rods.** The 5th-order polynomial fitted to Figure B.1 gave k = 9.59 at 68 rods, 10.73 at 72 and 72.9 at 100, where the chart reads 9.07, 9.16 and ends at 72 rods. The package now uses monotone piecewise cubic interpolation of the digitised points, with a point added at 72 rods (9.2). For more than 72 rods it holds k at 9.2 and issues a warning. Case `S34-FigB1-k-N100-beyond-chart` guards this.
2. **The `Network` solver depends on discretisation size and does not converge.** A single 3.6 m rod comes out 9 % above Formula R1 at Δ = 0.2 m and 24 % above at 0.4 m. The error is smallest near Δ = 0.05 m and grows again below that: +2.5 % at 0.025 m. Each segment is modelled as a disc of equal area, so the effective conductor radius changes with Δ. Grid results at Δ = 0.2 m agree with EPRI to within 0.2 % for resistance and 6 % for mesh voltage. The rod case `NUM-rod-vs-S34-R1` now runs at Δ = 0.1 m (+1.9 %). At 0.2 m it failed the ±5 % tolerance. See Discretisation guidance below.
3. **ENA S34 is inconsistent about the parameter b in Formula R6.** Appendix B defines b = w/π, and the package follows it. Worked examples 6.1.1 and 6.5.1 substitute b = 0.01 m directly. To reproduce the worked examples for a round conductor of diameter d, pass `w = pi * d`. Passing `w = d` gives R_E 5.7 % lower (1.53 Ω instead of 1.62 Ω).
4. **Mesh voltage depends on how a round conductor is turned into a strip.** With strip width 2d (equivalent radius w/4) the IEEE Example 1 mesh voltage is +5.9 % against EPRI. With the equal-area width π·d/2 it is +12.96 %.
5. **`max_current_density` uses the IS 3043 coefficient 7.57e3.** ENA S34 Formula C2 implies 1e3·√57.7 = 7.596e3, so the package's limit is 0.34 % lower (conservative). All TS 41-24 Table 8 ratings still agree to within 1 %.
6. **`e_mesh_step_grid` always uses IEEE 80 Eq. (91) for L_M, with K_ii = 1, whenever rods are present.** That form assumes rods at the corners and along the perimeter. For rods only in the interior, IEEE 80 specifies Eq. (90).
7. **`Network.generate_model` can raise `ZeroDivisionError` at small discretisation sizes** (seen for a rod at Δ = 0.00625 m). This happens when an evaluation point lies in an element's plane inside its radius. `generate_model_fast` handles the same case through `arctan(inf)`.

## Discretisation guidance

`generate_model_fast` replaces each segment with a flat circular disc of equal area. The error therefore depends on the step Δ relative to the conductor size, not on Δ alone. Refining Δ does not converge to the exact value: below the optimum the error grows again. The sweeps below compare a single electrode with a closed-form reference. Scripts are not kept in the repository; repeat them with `Network.add_rod`, `add_strip` or `add_plate` and the formulas named.

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

