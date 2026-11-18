# API reference

This reference documents the public names exported by `earthing` (the module's `__all__`) and the main methods of `Network`. Unless noted otherwise, dimensions are in metres, soil resistivity is in ohm-metres, current is in amperes, and voltage is in volts.

> **Important:** The equations are implemented in the package source and are associated with the standards named in the function docstrings. This reference describes software inputs and outputs; it does not certify compliance or validate a particular design.

## Public exports

```python
from earthing import (
    Network,
    resistance_plate, resistance_pipe, resistance_strip,
    resistance_grid, resistance_grid_with_rods,
    e_step_50, e_touch_50, e_step_70, e_touch_70,
    e_mesh_step, e_mesh_step_grid,
    e_touch_external, e_touch_fence, e_touch_fence_peripheral,
    e_step_external, surface_potential_hemisphere,
    surface_potential_rod, surface_potential_grid,
    contour_distance, touch_limit_ts4124, step_limit_ts4124,
    fault_current, max_current_density, current_ratio,
    resistance_parallel, rectangle,
)
```

## Resistance calculations

### `resistance_plate(rho, A, h)`

Plate-electrode resistance calculation (ENA EREC S34).

- `rho` (`float`): soil resistivity, ohm-m.
- `A` (`float`): plate area, m².
- `h` (`float`): burial depth, m.
- Returns: resistance, ohms (`float`).

### `resistance_pipe(rho, L, d)`

Pipe-electrode resistance calculation (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `L` (`float`): pipe length, m.
- `d` (`float`): pipe diameter, m.
- Returns: resistance, ohms (`float`).

### `resistance_strip(rho, L, w, t, method='IS3043', thickness=0.0)`

Resistance of a straight horizontal strip or round conductor.

- `rho` (`float`): soil resistivity, ohm-m.
- `L` (`float`): strip length, m.
- `w` (`float`): burial depth, m.
- `t` (`float`): strip width, or twice the diameter for a round conductor, m.
- `method` (`str`, optional):
  - `'IS3043'` (default) or `'IS3043:1987'`: IS 3043:1987, ρ/(2πL)·ln(2L²/(w·t)). Kept as the default for compatibility.
  - `'IS3043:2018'`: IS 3043:2018 clause 14.2.3, ρ/(2πL)·ln(4L/t). Ignores `w`. Identical to `'S34'`.
  - `'S34'`: ENA EREC S34 Formula R7, ρ/(2πL)·ln(2L/d) with d = t/2 (half-width of a tape or diameter of a round conductor). Ignores `w`; valid for burial depths of 0.5–0.6 m and L ≤ 60√ρ.
  - `'IEEE142'`: Dwight's formula for a buried horizontal strip, IEEE Std 142-2007 Table 4-5, with width `t` and thickness `thickness` at depth `w`.
- `thickness` (`float`, optional): strip thickness, m; used by `'IEEE142'` only (should be less than t/8).
- Returns: resistance, ohms (`float`).

### `resistance_grid(rho, A, L, h=0.5)`

Ground-grid resistance calculation (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `A` (`float`): area occupied by the grid, m².
- `L` (`float`): total buried conductor length, m.
- `h` (`float`, optional): grid burial depth, m; default `0.5`.
- Returns: resistance, ohms (`float`).

### `resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w, b=None)`

Grid resistance including vertical rods (ENA EREC S34).

- `rho` (`float`): soil resistivity, ohm-m.
- `A` (`float`): grid area, m².
- `L_E` (`float`): total horizontal-electrode length, m.
- `L_R` (`float`): rod length used by the equation, m.
- `N` (`int`): number of rods.
- `S` (`float`): rod separation, m.
- `d` (`float`): rod diameter, m.
- `w` (`float`): tape width, m. Used as b = w/π when `b` is not given.
- `b` (`float`, optional): equivalent diameter of the grid conductor in the R12 term, m. Overrides `w`.
- Returns: combined resistance, ohms (`float`).

ENA EREC S34 defines `b` inconsistently. Appendix B.1.7 prints b = w/π, which is the default. Appendix A and worked examples 6.1.1 and 6.5.1 use the conductor's equivalent diameter: b = d for a round conductor, or perimeter/π for a tape (as in Formula P1). Pass `b` to reproduce the worked examples.

Factor `k` is interpolated (monotone piecewise cubic) from points digitised from ENA EREC S34 Figure B.1, which covers 0–72 rods. For `N` greater than 72, `k` is held at the chart end value of 9.2 and a `UserWarning` is issued.

## Touch, step, and current calculations

### `e_step_50(rho, rho_s, h_s, t_s)` and `e_touch_50(rho, rho_s, h_s, t_s)`

Permissible step and touch potential for a 50 kg body (IEEE 80 Eqs. 29 and 32). Parameters are the same as `e_step_70`; C_s is from IEEE 80 Eq. 27. Returns the limit in volts (`float`).

### `e_step_70(rho, rho_s, h_s, t_s)`

Permissible step-potential calculation (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `rho_s` (`float`): surface-layer resistivity, ohm-m.
- `h_s` (`float`): surface-layer thickness, m.
- `t_s` (`float`): shock duration, s.
- Returns: step-potential limit, V (`float`).

### `e_touch_70(rho, rho_s, h_s, t_s)`

Permissible touch-potential calculation (IEEE 80). Parameters are the same as `e_step_70`; returns a touch-potential limit in volts (`float`).

### `e_mesh_step(rho, Lc, Lp, A, Lx, Ly, D, Lr, Nr, d, h, Ig, Dm=None, rods_on_perimeter=True)`

Mesh voltage and step voltage for square, rectangular, L-shaped and other irregular grids (IEEE 80 Eqs. 80–94).

- `rho` (`float`): soil resistivity, ohm-m.
- `Lc` (`float`): total length of horizontal grid conductor, m.
- `Lp` (`float`): peripheral length of the grid, m.
- `A` (`float`): grid area, m².
- `Lx`, `Ly` (`float`): maximum grid dimensions in X and Y, m.
- `D` (`float`): spacing between parallel conductors, m.
- `Lr` (`float`): length of each rod, m.
- `Nr` (`int`): number of rods.
- `d` (`float`): grid-conductor diameter, m.
- `h` (`float`): grid burial depth, m.
- `Ig` (`float`): grid current, A.
- `Dm` (`float`, optional): maximum distance between any two points on the grid, m. Defaults to √(Lx² + Ly²), so n_d = 1, as for square, rectangular and L-shaped grids.
- `rods_on_perimeter` (`bool`, optional): `True` (default) for rods in the corners and along the perimeter: K_ii = 1 and L_M from Eq. 91. `False` for no rods or a few rods inside the grid only: K_ii from Eq. 82 and L_M = L_C + L_R from Eq. 90.
- Returns: a two-item tuple `(Em, Es)`, mesh voltage and step voltage in volts.

### `e_mesh_step_grid(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig, rods_on_perimeter=True)`

Mesh voltage and step voltage of a rectangular grid (IEEE 80). Derives `Lc`, `Lp`, `A` and `D` from the grid dimensions and calls `e_mesh_step`.

- `rho` (`float`): soil resistivity, ohm-m.
- `Lx`, `Ly` (`float`): grid dimensions in X and Y, m.
- `Lr` (`float`): average rod length, m.
- `Nx`, `Ny` (`int`): number of grid conductors in the X and Y directions.
- `Nr` (`int`): number of earth rods.
- `d` (`float`): earth-conductor diameter, m.
- `h` (`float`): grid burial depth, m.
- `Ig` (`float`): grid current, A.
- `rods_on_perimeter` (`bool`, optional): see `e_mesh_step`.
- Returns: a two-item tuple `(Em, Es)`, mesh voltage and step voltage in volts.

## Surface potentials (ENA EREC S34 Appendix B.3)

Formula P4 of S34 is the IEEE 80 mesh voltage; use `e_mesh_step` or `e_mesh_step_grid`.

### `e_touch_external(rho, I_E, L_T, L_p, h, d, D, n_A, n_B)`

External touch potential 1 m beyond the edge of the grid electrode (Formula P1).

- `I_E` (`float`): current passing to ground through the electrode, A.
- `L_T` (`float`): total buried conductor length, including connected rods, m.
- `L_p` (`float`): perimeter conductor length, including connected rods, m.
- `h` (`float`): grid depth, m.
- `d` (`float`): equivalent conductor diameter (conductor perimeter / π), m.
- `D` (`float`): average spacing between parallel grid conductors, m.
- `n_A`, `n_B` (`int`): number of parallel grid conductors in each direction.
- Returns: touch potential, V.

### `e_touch_fence(rho, I_E, L_T, L_p, h, d, D, n_A, n_B)`

Touch potential at a separately earthed fence 2 m from the grid (Formula P2, k_f = 0.26·k_e). Parameters are the same as `e_touch_external`. The divisor is `L_T`, as in S34 worked example 6.1.5; Formula P2 as printed divides by `L_p`.

### `e_touch_fence_peripheral(rho, I_E, L_T, L_p, h, d, S)`

Touch potential at a fence with a buried peripheral conductor 1 m outside it (Formula P3, as printed). `S` is the distance between the outermost grid conductor and the next parallel conductor, m. S34 has no worked example for P3, so this function is not checked against a published value.

### `e_step_external(rho, I_E, R_E, x)`

Step potential over 1 m on the outside edge of a grid (Formula P5). `R_E` is the electrode resistance, ohms; `x` is the distance from the grid centre, m. Returns V.

### `surface_potential_hemisphere(rho, I, x)`, `surface_potential_rod(rho, I, L_R, x)`, `surface_potential_grid(rho, I, R_E, x)`

Surface potential at distance `x` (m) from a hemisphere, a vertical rod of length `L_R`, or a buried grid of resistance `R_E` (Formulae P6.1, P6.2, P6.3). `I` is the injected current, A. Returns V.

### `contour_distance(A, V_x, U_E)`

Distance from the edge of a grid of area `A` (m²) to the surface potential contour `V_x` (V), for an earth potential rise `U_E` (V) (Formula P7). Raises `ValueError` if `V_x` is not lower than `U_E`. Returns m.

## Permissible potentials (ENA TS 41-24)

### `touch_limit_ts4124(t, surface='shoes_chippings_75', interpolate='conservative')`

Permissible touch potential from ENA TS 41-24 Table 1, V.

- `t` (`float`): fault clearance time, s.
- `surface` (`str`): `'bare_feet'`, `'shoes_soil'` (shoes on soil or outdoor concrete), `'shoes_chippings_75'`, `'shoes_chippings_150'` (or dry concrete), or `'hand_to_hand'`.
- `interpolate` (`str`): between tabulated times, `'conservative'` returns the value at the next longer tabulated time; `'linear'` interpolates linearly. Times of 10 s or more return the ≥10 s value. Times below 0.1 s return the 0.1 s value with a `UserWarning`.

### `step_limit_ts4124(t, surface='shoes_chippings_75', interpolate='conservative')`

Permissible step potential from ENA TS 41-24 Table 2, V. Arguments as `touch_limit_ts4124`, except that `'hand_to_hand'` is not available. Returns `math.inf` where the table states the limit cannot foreseeably be exceeded (25 kV or more).

### `fault_current(Un, c, Z)`

Fault-current calculation (IEC 60909).

- `Un` (`float`): nominal line-to-line voltage, V.
- `c` (`float`): voltage factor.
- `Z` (`complex`): complex impedance, ohms.
- Returns: fault-current magnitude, A (`float`).

### `max_current_density(rho, t, method='IS3043')`

Maximum electrode current density.

- `rho` (`float`): soil resistivity, ohm-m.
- `t` (`float`): loading time, s.
- `method` (`str`, optional): `'IS3043'` (default), IS 3043:2018 clause 15.3, 7.57×10³/√(ρt); or `'S34'`, ENA EREC S34 Formula C2, 10³·√(57.7/(ρt)). The S34 value is 0.34 % higher and is the one used for ENA TS 41-24 Table 8.
- Returns: current density, A/m² (`float`).

### `current_ratio(rho, R, C, Un, A, L)`

Current-ratio calculation using the C-factor method (ENA EREC S34).

- `rho` (`float`): soil resistivity, ohm-m.
- `R` (`float`): combined source and destination resistance, ohms.
- `C` (`float`): C factor.
- `Un` (`float`): system voltage, kV.
- `A` (`float`): cable cross-sectional area, mm².
- `L` (`float`): cable length, km.
- Returns: calculated current ratio (dimensionless `float`).

## Utility functions

### `resistance_parallel(*res)`

Calculates equivalent resistance for parallel resistances.

- `*res` (`float`): one or more resistance values in ohms.
- Returns: equivalent resistance in ohms (`float`).

### `rectangle(loc, x_l, y_l)`

Creates a closed rectangular polygon.

- `loc` (`tuple[float, float]`): starting `(x, y)` coordinate.
- `x_l` (`float`): rectangle length along X, m.
- `y_l` (`float`): rectangle width along Y, m.
- Returns: tuple of five `(x, y)` vertices; the final vertex repeats the first to close the polygon.

## Network class

### `Network(rho, Ig)`

Creates a numerical earth-network model.

- `rho` (`float`): soil resistivity, ohm-m.
- `Ig` (`float` or sequence of current values): current injection for the first subnet, A.
- Returns: a `Network` instance.

### Geometry and model methods

| Method | Parameters | Purpose / result |
| --- | --- | --- |
| `clear_model()` | none | Clears discretised geometry, matrices, and calculated solution state. |
| `add_subnet(Ig)` | `Ig`: current injection, A | Adds a subnet with its own current injection. |
| `add_strip(loc_start, loc_end, w)` | start/end 3D coordinates; strip width `w`, m | Adds a horizontal or sloping strip electrode between two coordinates. |
| `add_rod(loc, radius, length)` | top coordinate; rod radius and length, m | Adds a vertical rod extending down from `loc`. |
| `add_mesh(loc, Lx, Ly, Nx, Ny, w)` | origin; X/Y dimensions; conductor counts; strip width | Adds a uniform rectangular mesh. |
| `add_plate(loc, w, h, n_cap=[1,0,0], h_cap=[0,0,1])` | position; width/height; optional normal and height direction vectors | Adds a plate electrode. |
| `mesh_geometry(desc_size=0.2)` | discretisation size, m | Discretises the electrode geometry into calculation elements. |
| `generate_model(desc_size=0.25)` | discretisation size, m | Builds the system of equations. |
| `generate_model_fast(desc_size=0.25)` | discretisation size, m | Builds a faster approximate model using circular-plate discretisation. |
| `solve_model()` | none | Solves the generated system and stores the solution on the object. |

The geometry methods add elements to the current subnet. Call `add_subnet()` before adding elements that belong to another current-injection subnet.

### Results and potential methods

| Method | Parameters | Purpose / result |
| --- | --- | --- |
| `get_resistance()` | none | Returns the calculated resistance for each subnet, rounded to 3 decimal places. |
| `gpr()` | none | Returns ground potential rise for each subnet, rounded to 3 decimal places. |
| `get_point_potential(loc)` | `loc`: 3D coordinate, m | Calculates potential at the specified point. |
| `solve_surface_potential(grid=(20,20), xlim=(-20,20), ylim=(-20,20))` | grid point counts and X/Y limits, m | Calculates surface potential on a regular grid. |
| `solve_surface_potential_fast(grid=(20,20), xlim=(-20,20), ylim=(-20,20), save_results=True)` | grid point counts, X/Y limits, save flag | Faster approximate surface-potential calculation. |
| `mesh_voltage(polygon_points, mesh_no=10, plot=False, plot_type='fill', levels=10, grid_spacing=1, subnet=0)` | polygon vertices; sampling density; plot controls; subnet index | Evaluates mesh voltage within the polygon. |
| `step_voltage(polygon_points, mesh_no=14, step_size=1, plot=False, plot_type='fill', levels=10, grid_spacing=1)` | polygon vertices; sampling density; step size; plot controls | Evaluates step voltage within the polygon. |

Surface-potential methods calculate values on a grid and store plotting data on the object. The fast methods use an approximation; compare them with the standard methods for representative cases. Mesh and step voltage methods accept polygon corner coordinates in metres.

### Plotting methods

| Method | Purpose |
| --- | --- |
| `plot_surface_potential(xlim=(-20,20), ylim=(-20,20), plot_type='fill', levels=10, grid_spacing=1, plot_data=None, title='Surface potential distribution')` | Plots surface potential data, either stored on the object or supplied through `plot_data`. |
| `plot_geometry_3d(xlim=(-20,20), ylim=(-20,20), zlim=(-5,5), ground=True, ground_pot=False, current_distribution=False, normal=False, normal_scale=0.5, title='Problem geometry')` | Displays 3D electrode geometry and optional ground-potential, current-distribution, or element-normal information. |

Plotting methods display Matplotlib figures; they are not intended to return numerical design results.

## Implementation classes

The following classes exist in `earthing.__init__` but are not included in the module's `__all__`. They are implementation details rather than the documented public interface:

- `DiscreteElement`, `DiscreteElementCylindrical`, `DiscreteElementPlate` (old spellings `Descrete*` kept as deprecated aliases): numerical potential-coefficient elements.
- `NetworkElement`, `NetworkElementPlate`, `NetworkElementStrip`, `NetworkElementPipe`: electrode primitives and their discretisation helpers.

Their methods may change without notice. Prefer the exported functions and `Network` class.

## Exceptions and calculation state

Several `Network` methods require a generated or solved model. Generate the model before solving it, and solve it before requesting resistance, GPR, point potential, or solution-dependent plots. The current implementation raises exceptions when required geometry or solution state is missing.

## References

- [Project README](../README.md)
- [User guide](user-guide.md)
- [Source module](../earthing/__init__.py)
- [Worked notebooks](../)
