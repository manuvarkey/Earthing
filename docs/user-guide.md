# User guide

This guide introduces the `earthing` Python package and shows how to use its calculation functions and numerical network model.

> **Engineering notice:** Results are calculation aids, not a substitute for a competent earthing-system design review. Confirm the applicable edition and requirements of IEEE 80, ENA EREC S34, IS 3043, IEC 60909, and local regulations. Verify all inputs, assumptions, soil models, fault-current data, and safety limits independently before using results for a real installation.

## 1. Installation

The package is published on PyPI:

```bash
python -m pip install earthing
```

The package metadata declares Python 3.6 or newer, NumPy 1.22 or newer, and Matplotlib 3.5 or newer.

Check the installation:

```python
import earthing

print(earthing.__all__)
```

The package's public exports are listed in [API reference](api-reference.md).

## 2. Units and input conventions

Use a consistent SI unit system unless a function's description explicitly specifies otherwise.

| Quantity | Expected unit |
| --- | --- |
| Soil resistivity (`rho`) | ohm-m |
| Length, depth, spacing, diameter | m |
| Area | m² |
| Current | A |
| Voltage | V |
| Time | s |
| Cable cross-sectional area in `current_ratio` | mm² |
| Cable length in `current_ratio` | km |
| System voltage in `current_ratio` | kV |
| Fault impedance in `fault_current` | ohm |

Coordinates passed to `Network` methods are three-dimensional `(x, y, z)` positions in metres. Use the same coordinate reference and sign convention throughout a model.

## 3. Quick calculation examples

Import the functions you need:

```python
from earthing import (
    resistance_grid,
    e_touch_70,
    e_step_70,
    fault_current,
)

# Illustrative inputs only
rho = 100.0       # soil resistivity, ohm-m
grid_area = 400.0 # m²
buried_length = 120.0 # m
grid_depth = 0.5  # m

r_grid = resistance_grid(rho, grid_area, buried_length, grid_depth)

# Prospective fault current: 11 kV line-to-line, voltage factor 1.1
# and a complex Thevenin impedance in ohms.
i_fault = fault_current(11_000, 1.1, 0.1 + 0.8j)

# Permissible touch and step limits (illustrative surface layer data)
e_touch = e_touch_70(rho, 3000.0, 0.15, 1.0)
e_step = e_step_70(rho, 3000.0, 0.15, 1.0)

print(r_grid, i_fault, e_touch, e_step)
```

The functions return numerical values. The formulas and standard references are summarized in the API reference. Check the source and applicable standard before relying on a result.

## 4. Numerical earth-network workflow

The `Network` class models electrode geometry, discretises it into calculation elements, solves the network equations, and evaluates potentials.

A typical workflow is:

1. Create a network with soil resistivity and injected grid current.
2. Add electrode geometry (strips, rods, meshes, plates, or subnets).
3. Generate the numerical model.
4. Solve the model.
5. Read resistance and ground-potential-rise results.
6. Calculate point or surface potentials and, where required, mesh and step voltages.

### Example: rectangular grid

```python
from earthing import Network

soil_resistivity = 100.0  # ohm-m
grid_current = 1000.0     # A

network = Network(soil_resistivity, grid_current)

# Start point, X/Y dimensions, number of conductors in each direction,
# and conductor strip width (all dimensions in metres).
network.add_mesh(
    loc=(0.0, 0.0, -0.5),
    Lx=20.0,
    Ly=20.0,
    Nx=5,
    Ny=5,
    w=0.05,
)

# Discretisation size controls the numerical model resolution.
network.generate_model(desc_size=0.5)
network.solve_model()

print("Grid resistance (ohm):", network.get_resistance())
print("Ground potential rise (V):", network.gpr())
```

The example demonstrates the call sequence; it is not a validated design or a recommended grid layout. Refine the discretisation and compare results for convergence before using a model.

### Add individual electrodes

```python
from earthing import Network

network = Network(rho=100.0, Ig=500.0)

# Horizontal strip between two 3D coordinates.
network.add_strip(
    loc_start=(0.0, 0.0, -0.5),
    loc_end=(10.0, 0.0, -0.5),
    w=0.05,
)

# Vertical rod: loc is the top coordinate; length extends downwards.
network.add_rod(loc=(5.0, 5.0, 0.0), radius=0.0125, length=3.0)

# Plate: position, width, height; optional normal and height direction.
network.add_plate(loc=(2.0, 2.0, -1.0), w=0.6, h=0.6)

network.generate_model()
network.solve_model()
```

### Multiple current-injection subnets

Use `add_subnet(Ig)` to add another subnet and its current injection before adding that subnet's geometry. The `+` operator also combines two `Network` objects. Each subnet is assigned its own potential in the solved model.

## 5. Potential calculations and plots

After `generate_model()` and `solve_model()`:

- `get_point_potential(loc)` evaluates potential at one coordinate.
- `solve_surface_potential(...)` calculates surface potentials on a regular grid.
- `solve_surface_potential_fast(...)` uses a faster approximate element-potential assumption.
- `mesh_voltage(polygon_points, ...)` evaluates mesh voltage in a polygon.
- `step_voltage(polygon_points, ...)` evaluates step voltage in a polygon.
- `plot_surface_potential(...)` plots previously calculated surface-potential data.
- `plot_geometry_3d(...)` displays the electrode geometry and optional solution information.

The plotting methods use Matplotlib. Surface-potential calculation methods populate data used by the plotting routines; see the API reference for arguments and defaults.

## 6. Choosing a calculation method

- Use the standalone resistance and safety-limit functions for quick calculations based on their stated equations.
- Use `Network` when electrode geometry and spatial potential distribution matter.
- Use `generate_model_fast()` or `solve_surface_potential_fast()` only when their approximation is suitable for the engineering question. Compare against the standard method for representative cases.
- Use mesh and step voltage calculations to investigate local voltage gradients in the selected polygon.

## 7. Existing worked examples

The repository includes Jupyter notebooks that demonstrate formula-based substation calculations and numerical earth-network simulations. Open them from the repository:

- [Earth design using formula](../Earth%20design%20using%20formula.ipynb)
- [Earth design simulation](../Earth%20design%20simulation.ipynb)
- [Notebook and benchmark tests](../tests/)

The notebooks are useful examples, but review their assumptions and input data before reusing them.

## 8. Troubleshooting

- **Model generation reports no geometry:** add at least one electrode to the current subnet before calling `generate_model()`.
- **Solution requested before model generation:** call `generate_model()` before `solve_model()`.
- **Potential or plotting method reports model not solved:** call `solve_model()` first.
- **Unexpected or unstable values:** check units, coordinate positions, geometry dimensions, current injections, soil resistivity, and discretisation size. Compare results at more than one discretisation size.
- **Missing plotting dependency:** install the declared dependencies with `python -m pip install -U numpy matplotlib`.

## 9. API reference

See [API reference](api-reference.md) for the exported calculation functions, the `Network` class, method signatures, inputs, and return values.
