# API reference

This reference documents the public names exported by `earthing` (the module's `__all__`) and the main methods of `Network`. Unless noted otherwise, dimensions are in metres, soil resistivity is in ohm-metres, current is in amperes, and voltage is in volts.

> **Important:** The equations are implemented in the package source and are associated with the standards named in the function docstrings. This reference describes software inputs and outputs; it does not certify compliance or validate a particular design.

## Public exports

```python
from earthing import (
    Network,
    resistance_plate, resistance_pipe, resistance_strip,
    resistance_grid, resistance_grid_with_rods,
    e_step_70, e_touch_70, e_mesh_step_grid,
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

### `resistance_strip(rho, L, w, t)`

Strip-electrode resistance calculation (IS 3043).

- `rho` (`float`): soil resistivity, ohm-m.
- `L` (`float`): strip length, m.
- `w` (`float`): burial depth, m.
- `t` (`float`): strip thickness; the source docstring also describes using twice the diameter for a pipe, m.
- Returns: resistance, ohms (`float`).

### `resistance_grid(rho, A, L, h=0.5)`

Ground-grid resistance calculation (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `A` (`float`): area occupied by the grid, m².
- `L` (`float`): total buried conductor length, m.
- `h` (`float`, optional): grid burial depth, m; default `0.5`.
- Returns: resistance, ohms (`float`).

### `resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w)`

Grid resistance including vertical rods (ENA EREC S34).

- `rho` (`float`): soil resistivity, ohm-m.
- `A` (`float`): grid area, m².
- `L_E` (`float`): total horizontal-electrode length, m.
- `L_R` (`float`): rod length used by the equation, m.
- `N` (`int`): number of rods.
- `S` (`float`): rod separation, m.
- `d` (`float`): rod diameter, m.
- `w` (`float`): tape width, m.
- Returns: combined resistance, ohms (`float`).

## Touch, step, and current calculations

### `e_step_70(rho, rho_s, h_s, t_s)`

Permissible step-potential calculation (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `rho_s` (`float`): surface-layer resistivity, ohm-m.
- `h_s` (`float`): surface-layer thickness, m.
- `t_s` (`float`): shock duration, s.
- Returns: step-potential limit, V (`float`).

### `e_touch_70(rho, rho_s, h_s, t_s)`

Permissible touch-potential calculation (IEEE 80). Parameters are the same as `e_step_70`; returns a touch-potential limit in volts (`float`).

### `e_mesh_step_grid(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig)`

Calculates grid mesh voltage and step voltage (IEEE 80).

- `rho` (`float`): soil resistivity, ohm-m.
- `Lx`, `Ly` (`float`): grid dimensions in X and Y, m.
- `Lr` (`float`): average rod length, m.
- `Nx`, `Ny` (`int`): number of grid conductors in the X and Y directions.
- `Nr` (`int`): number of earth rods.
- `d` (`float`): earth-conductor diameter, m.
- `h` (`float`): grid burial depth, m.
- `Ig` (`float`): grid current, A.
- Returns: a two-item tuple `(Em, Es)`, mesh voltage and step voltage in volts.

### `fault_current(Un, c, Z)`

Fault-current calculation (IEC 60909).

- `Un` (`float`): nominal line-to-line voltage, V.
- `c` (`float`): voltage factor.
- `Z` (`complex`): complex impedance, ohms.
- Returns: fault-current magnitude, A (`float`).

### `max_current_density(rho, t)`

Maximum electrode current-density calculation (IS 3043).

- `rho` (`float`): soil resistivity, ohm-m.
- `t` (`float`): loading time, s.
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

- `DescreteElement`, `DescreteElementCylindrical`, `DescreteElementPlate`: numerical potential-coefficient elements.
- `NetworkElement`, `NetworkElementPlate`, `NetworkElementStrip`, `NetworkElementPipe`: electrode primitives and their discretisation helpers.

Their methods may change without notice. Prefer the exported functions and `Network` class.

## Exceptions and calculation state

Several `Network` methods require a generated or solved model. Generate the model before solving it, and solve it before requesting resistance, GPR, point potential, or solution-dependent plots. The current implementation raises exceptions when required geometry or solution state is missing.

## References

- [Project README](../README.md)
- [User guide](user-guide.md)
- [Source module](../earthing/__init__.py)
- [Worked notebooks](../)
