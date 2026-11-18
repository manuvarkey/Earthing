#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation cases for the ``earthing`` package.

Every case compares a value computed by the package against a value taken
from one of the reference standards:

* IEEE Std 80-2000, "IEEE Guide for Safety in AC Substation Grounding"
* ENA EREC S34 Issue 2 (2018), "A guide for assessing the rise of earth
  potential at electrical installations"
* ENA TS 41-24 Issue 2 (2018), "Guidelines for the design, installation,
  testing and maintenance of main earthing systems in substations"

Reference values are transcribed from the worked examples and tables of those
documents; clause, equation and printed page numbers are given for each case
so that the source can be checked. The standards themselves are not needed at
run time.

Case kinds
----------
``closed-form``  Closed-form design functions (resistance_*, e_*, fault_current,
                 max_current_density, current_ratio, ...). Tolerances are tight
                 and cover only rounding in the published values.
``numerical``    The boundary-discretisation solver (``Network``) compared with
                 benchmark values from the standards. Engineering tolerances
                 apply (resistance 5 %, voltages 10 %).

A case whose ``rtol`` and ``atol`` are both ``None`` is informational: it is
computed and reported but never passes or fails.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from dataclasses import dataclass
from math import asin, log, pi, sqrt
from typing import Callable, List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")  # the package imports pyplot

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import earthing as e  # noqa: E402

IEEE80 = "IEEE Std 80-2000"
S34 = "ENA EREC S34:2018"
TS4124 = "ENA TS 41-24:2018"
IEEE142 = "IEEE Std 142-2007"

CLOSED = "closed-form"
NUMERICAL = "numerical"

# Discretisation used for numerical benchmarks. 0.2 m is the value used in the
# project's own benchmark notebooks (tests/*.ipynb).
DELTA = 0.2
# Discretisation for the single-rod case. Rod error is smallest at a step of
# about 3-4 rod diameters (see validation/README.md, "Discretisation
# guidance"); 0.1 m (about 6 d for a 16 mm rod) gives about +2 %.
DELTA_ROD = 0.1
# Finer rod discretisation (about 3 rod diameters, the minimum-error step) for
# cases that test other solver features against closed-form references.
DELTA_ROD_FINE = 0.05

# Tolerances
RTOL_ROUNDING = 0.01      # published values rounded to 3-4 significant figures
RTOL_NUM_RESISTANCE = 0.05
RTOL_NUM_VOLTAGE = 0.10


@dataclass
class Case:
    id: str
    standard: str
    clause: str
    description: str
    function: str
    compute: Callable[[], float]
    reference: float
    unit: str
    rtol: Optional[float] = RTOL_ROUNDING
    atol: Optional[float] = None
    kind: str = CLOSED
    note: str = ""

    @property
    def informational(self) -> bool:
        return self.rtol is None and self.atol is None

    def allowed_error(self) -> float:
        allowed = 0.0
        if self.rtol is not None:
            allowed = max(allowed, self.rtol * abs(self.reference))
        if self.atol is not None:
            allowed = max(allowed, self.atol)
        return allowed

    def tolerance_text(self) -> str:
        if self.informational:
            return "info"
        parts = []
        if self.rtol is not None:
            parts.append("+/-{:g} %".format(self.rtol * 100))
        if self.atol is not None:
            parts.append("+/-{:g} {}".format(self.atol, self.unit).strip())
        return " / ".join(parts)


@dataclass
class Result:
    case: Case
    computed: Optional[float]
    status: str          # PASS, FAIL, INFO or ERROR
    error: str = ""

    @property
    def deviation(self) -> Optional[float]:
        if self.computed is None or self.case.reference == 0:
            return None
        return (self.computed - self.case.reference) / abs(self.case.reference)


def evaluate(case: Case) -> Result:
    try:
        with warnings.catch_warnings():
            # generate_model_fast divides by zero on purpose (arctan(inf))
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(case.compute())
    except Exception as exc:  # report, don't abort the whole run
        return Result(case, None, "ERROR", "{}: {}".format(type(exc).__name__, exc))
    if case.informational:
        return Result(case, value, "INFO")
    ok = abs(value - case.reference) <= case.allowed_error()
    return Result(case, value, "PASS" if ok else "FAIL")


# ---------------------------------------------------------------------------
# Reference formulas written directly from the standards. They are used to
# check that the package implements the formula as printed, and to back out
# intermediate quantities (e.g. factor k) from package results.
# ---------------------------------------------------------------------------

def ieee80_eq55_single_rod(rho, L_r, diameter):
    """IEEE 80 Eq. (55) (Schwarz rod bed) with n_R = 1; 2b = rod diameter."""
    b = diameter / 2
    return rho / (2 * pi * L_r) * (log(4 * L_r / b) - 1)


def s34_r1_rod(rho, L_r, d):
    """ENA EREC S34 B.1.2 Formula R1."""
    return rho / (2 * pi * L_r) * (log(8 * L_r / d) - 1)


def s34_r2_plate(rho, A, h):
    """ENA EREC S34 B.1.3 Formula R2."""
    r = sqrt(A / pi)
    return rho / (8 * r) * (1 + r / (2.5 * h + r))


def s34_r4_grid(rho, A, L_E):
    """ENA EREC S34 B.1.5 Formula R4."""
    return rho / (4 * sqrt(A / pi)) + rho / L_E


def s34_r7_strip(rho, L_H, d):
    """ENA EREC S34 B.1.8 Formula R7 (d = diameter or tape half-width)."""
    return rho / (2 * pi * L_H) * log(2 * L_H / d)


def s34_k_from_package(rho, A, L_E, L_R, N, S, d, w):
    """Back out the factor k used by ``resistance_grid_with_rods``.

    Formula R6 is inverted for R2 (the rod-group resistance), then Formula R5
    is inverted for k. R1, R12 and R_R do not depend on k, so this recovers
    exactly the k value the package used.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # N beyond Figure B.1
        R_E = e.resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w)
    R1 = s34_r4_grid(rho, A, L_E)
    R12 = R1 - rho / (pi * L_E) * (log(L_R / (w / pi)) - 1)
    R2 = (R12 ** 2 + R_E * R1 - 2 * R_E * R12) / (R1 - R_E)
    R_R = s34_r1_rod(rho, L_R, d)
    alpha = rho / (2 * pi * R_R * S)
    return (R2 * N / R_R - 1) / alpha


def ieee80_em_interior_rods(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig):
    """IEEE 80 Eqs. (80)-(90) for a rectangular grid whose rods are inside
    the grid only: K_ii from Eq. (82), L_M = L_C + L_R from Eq. (90)."""
    A = Lx * Ly
    L_C = Lx * Nx + Ly * Ny
    L_p = 2 * (Lx + Ly)
    D = (Lx / (Ny - 1) + Ly / (Nx - 1)) / 2
    n = (2 * L_C / L_p) * sqrt(L_p / (4 * sqrt(A)))  # n_c = n_d = 1
    K_ii = 1 / (2 * n) ** (2 / n)
    K_h = sqrt(1 + h / 1.0)
    K_m = 1 / (2 * pi) * (log(D ** 2 / (16 * h * d) + (D + 2 * h) ** 2 / (8 * D * d)
                              - h / (4 * d)) + K_ii / K_h * log(8 / (pi * (2 * n - 1))))
    K_i = 0.644 + 0.148 * n
    return rho * Ig * K_m * K_i / (L_C + Lr * Nr)


def ieee80_cs_limit(C_s, rho_s, t_s, foot, k):
    """IEEE 80 Eqs. (29)-(33) with a given C_s: foot = 6 (step) or 1.5
    (touch); k = 0.116 (50 kg) or 0.157 (70 kg)."""
    return (1000 + foot * C_s * rho_s) * k / sqrt(t_s)


def ieee142_dwight_strip(rho, length, a, b, depth):
    """IEEE Std 142-2007 Table 4-5, buried horizontal strip (length 2L,
    section a x b, depth s/2)."""
    L = length / 2
    s = 2 * depth
    return rho / (4 * pi * L) * (log(4 * L / a) + (a ** 2 - pi * a * b) / (2 * (a + b) ** 2)
                                 + log(4 * L / s) - 1 + s / (2 * L)
                                 - s ** 2 / (16 * L ** 2) + s ** 4 / (512 * L ** 4))


def s34_p6_3_grid(rho, I, R_E, x):
    """ENA EREC S34 B.3.7 Formula P6.3."""
    r = rho / (4 * R_E)
    return rho * I / (2 * pi * r) * asin(r / x)


def ts4124_jlimit(rho, t):
    """ENA EREC S34 B.2.2 Formula C2, in A/m^2 (TS 41-24 5.4.6.3 refers to it)."""
    return 1e-3 * sqrt(57.7 / (rho * t)) * 1e6


# ---------------------------------------------------------------------------
# Closed-form cases: IEEE Std 80-2000
# ---------------------------------------------------------------------------

def _ieee80_cases() -> List[Case]:
    cases = []

    # Annex B, Step 3 (p. 131-132): rho = 400, rho_s = 2500, h_s = 0.102 m,
    # t_s = 0.5 s. IEEE rounds Cs to 0.74 before evaluating the limits; the
    # package uses the unrounded Eq. (27) value (0.7429), hence +0.3 %.
    cs_note = ("IEEE rounds Cs = 0.7429 to 0.74 before substitution; the "
               "package keeps full precision (about +0.3 %).")
    cases.append(Case(
        "IEEE80-B1-Estep70", IEEE80, "Annex B.1 Step 3, Eqs. (27), (30), (B.6)",
        "Tolerable step voltage, 70 kg body", "e_step_70",
        lambda: e.e_step_70(400, 2500, 0.102, 0.5), 2686.6, "V", note=cs_note))
    cases.append(Case(
        "IEEE80-B1-Etouch70", IEEE80, "Annex B.1 Step 3, Eqs. (27), (33), (B.7)",
        "Tolerable touch voltage, 70 kg body", "e_touch_70",
        lambda: e.e_touch_70(400, 2500, 0.102, 0.5), 838.2, "V", note=cs_note))

    # Annex B, Step 2 (p. 130): symmetrical ground-fault current.
    # 3I0 = 3E / |Z1 + Z2 + Z0| with E = U/sqrt(3). With c = 1 and
    # Z = (Z1 + Z2 + Z0)/3, fault_current() gives the same expression.
    z_115 = (4.0 + 10.0j) * 2 + (10.0 + 40.0j)
    z_13 = (0.085 + 1.142j) * 2 + (0.034 + 1.014j)
    fc_note = "Evaluated as fault_current(U, c=1, Z=(Z1+Z2+Z0)/3)."
    cases.append(Case(
        "IEEE80-B1-3I0-115kV", IEEE80, "Annex B.1 Step 2, Eq. (B.1)",
        "Ground-fault current 3I0, 115 kV bus", "fault_current",
        lambda: e.fault_current(115e3, 1.0, z_115 / 3), 3180, "A",
        rtol=0.001, note=fc_note))
    cases.append(Case(
        "IEEE80-B1-3I0-13kV", IEEE80, "Annex B.1 Step 2, Eq. (B.1)",
        "Ground-fault current 3I0, 13 kV bus", "fault_current",
        lambda: e.fault_current(13e3, 1.0, z_13 / 3), 6814, "A",
        rtol=0.001, note=fc_note))

    # Grid resistance, Eq. (52) (Sverak), h = 0.5 m, rho = 400.
    for cid, clause, desc, A, L_T, ref in [
        ("IEEE80-B1-Rg", "Annex B.1 Step 5, Eq. (B.8)",
         "70 m x 70 m grid, no rods", 4900, 1540, 2.78),
        ("IEEE80-B2-Rg", "Annex B.2 Step 5, Eq. (B.20)",
         "70 m x 70 m grid, 20 rods x 7.5 m", 4900, 1690, 2.75),
        ("IEEE80-B3-Rg", "Annex B.3 Step 5, Eq. (B.26)",
         "63 m x 84 m grid, 38 rods x 10 m", 5292, 2039, 2.62),
        ("IEEE80-B4-Rg", "Annex B.4 Step 5, Eq. (B.33)",
         "L-shaped grid, 24 rods x 7.5 m", 4900, 1755, 2.74),
    ]:
        cases.append(Case(
            cid, IEEE80, clause, "Grid resistance: " + desc, "resistance_grid",
            (lambda A=A, L_T=L_T: e.resistance_grid(400, A, L_T, 0.5)),
            ref, "ohm"))

    # Mesh and step voltage, Eqs. (80)-(94). d = 0.01 m, h = 0.5 m, IG = 1908 A.
    km_note = ("IEEE rounds Km to two decimals before substitution, which "
               "accounts for differences of a few tenths of a percent.")
    cases.append(Case(
        "IEEE80-B1-Em", IEEE80, "Annex B.1 Step 8, Eqs. (80), (81), (89), (90)",
        "Mesh voltage: 70 m x 70 m grid, 11 x 11 conductors, no rods",
        "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(400, 70, 70, 0, 11, 11, 0, 0.01, 0.5, 1908)[0],
        1002.1, "V", note=km_note))
    cases.append(Case(
        "IEEE80-B2-Em", IEEE80, "Annex B.2 Step 8, Eqs. (80), (81), (91)",
        "Mesh voltage: 70 m x 70 m grid, 20 rods x 7.5 m", "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(400, 70, 70, 7.5, 11, 11, 20, 0.01, 0.5, 1908)[0],
        747.4, "V", note=km_note))
    cases.append(Case(
        "IEEE80-B2-Es", IEEE80, "Annex B.2 Step 8, Eqs. (92)-(94), (B.25)",
        "Step voltage: 70 m x 70 m grid, 20 rods x 7.5 m", "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(400, 70, 70, 7.5, 11, 11, 20, 0.01, 0.5, 1908)[1],
        548.9, "V"))
    cases.append(Case(
        "IEEE80-B3-Em", IEEE80, "Annex B.3 Step 8, Eqs. (84)-(89), (91)",
        "Mesh voltage: 63 m x 84 m grid (13 x 63 m + 10 x 84 m), 38 rods x 10 m",
        "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(400, 63, 84, 10, 13, 10, 38, 0.01, 0.5, 1908)[0],
        595.8, "V", note=km_note))

    # Example 4 (L-shaped grid, 24 perimeter rods), through the general
    # e_mesh_step: A = 4900 m2, L_C = 1575 m, L_p = 350 m, Lx = 70 m,
    # Ly = 105 m, D = 7 m. IEEE rounds n = 12.155 to 12.2, Km to 0.76,
    # Ks = 0.406 to 0.41 and Ki = 2.443 to 2.45.
    b4 = lambda: e.e_mesh_step(400, 1575, 350, 4900, 70, 105, 7, 7.5, 24,
                               0.01, 0.5, 1908)
    cases.append(Case(
        "IEEE80-B4-Em", IEEE80, "Annex B.4 Step 8, Eqs. (84)-(89), (91), (B.40)",
        "Mesh voltage: L-shaped grid (70 m x 105 m, A = 4900 m2), 24 rods x 7.5 m",
        "e_mesh_step", lambda: b4()[0], 761.1, "V", note=km_note))
    cases.append(Case(
        "IEEE80-B4-Es", IEEE80, "Annex B.4 Step 8, Eqs. (92)-(94), (B.42)",
        "Step voltage: L-shaped grid, 24 rods x 7.5 m", "e_mesh_step",
        lambda: b4()[1], 574.6, "V", rtol=0.015,
        note=("IEEE rounds Ks = 0.406 to 0.41 (+0.9 %) and Ki = 2.443 to "
              "2.45 (+0.3 %) before substitution.")))
    cases.append(Case(
        "IEEE80-B2-Em-general", IEEE80, "Eqs. (80)-(91)",
        "e_mesh_step with Example 2 inputs equals e_mesh_step_grid",
        "e_mesh_step",
        lambda: e.e_mesh_step(400, 1540, 280, 4900, 70, 70, 7, 7.5, 20,
                              0.01, 0.5, 1908)[0],
        e.e_mesh_step_grid(400, 70, 70, 7.5, 11, 11, 20, 0.01, 0.5, 1908)[0],
        "V", rtol=1e-9))

    # Rods inside the grid only: K_ii from Eq. (82), L_M from Eq. (90).
    # IEEE 80 has no worked example, so the formula is checked as printed.
    cases.append(Case(
        "IEEE80-Eq90-interior-rods", IEEE80, "Eqs. (82), (90)",
        "Mesh voltage: 70 m x 70 m grid, 9 interior rods x 7.5 m (formula identity)",
        "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(400, 70, 70, 7.5, 11, 11, 9, 0.01, 0.5, 1908,
                                   rods_on_perimeter=False)[0],
        ieee80_em_interior_rods(400, 70, 70, 7.5, 11, 11, 9, 0.01, 0.5, 1908),
        "V", rtol=1e-9))

    # 50 kg body limits, Eqs. (29) and (32). Clause 17 (fence case study):
    # rho = 60, rho_s = 3000, h_s = 0.076 m, t_s = 0.5 s. IEEE takes
    # Cs = 0.62 from Figure 11; Eq. (27), used by the package, gives 0.6355.
    cs50_note = ("IEEE reads Cs = 0.62 from Figure 11; the package uses "
                 "Eq. (27), Cs = 0.6355, hence about +2 %.")
    cases.append(Case(
        "IEEE80-17-Estep50", IEEE80, "Clause 17 fence case study, Eq. (29)",
        "Tolerable step voltage, 50 kg body", "e_step_50",
        lambda: e.e_step_50(60, 3000, 0.076, 0.5), 1995, "V", rtol=0.03,
        note=cs50_note))
    cases.append(Case(
        "IEEE80-17-Etouch50", IEEE80, "Clause 17 fence case study, Eq. (32)",
        "Tolerable touch voltage, 50 kg body", "e_touch_50",
        lambda: e.e_touch_50(60, 3000, 0.076, 0.5), 622, "V", rtol=0.03,
        note=cs50_note))
    cs_b1 = 1 - 0.09 * (1 - 400 / 2500) / (2 * 0.102 + 0.09)
    cases.append(Case(
        "IEEE80-Eq29-identity", IEEE80, "Eqs. (27), (29)",
        "Step limit, 50 kg: Example 1 inputs (formula identity)", "e_step_50",
        lambda: e.e_step_50(400, 2500, 0.102, 0.5),
        ieee80_cs_limit(cs_b1, 2500, 0.5, 6, 0.116), "V", rtol=1e-9))
    cases.append(Case(
        "IEEE80-Eq32-identity", IEEE80, "Eqs. (27), (32)",
        "Touch limit, 50 kg: Example 1 inputs (formula identity)", "e_touch_50",
        lambda: e.e_touch_50(400, 2500, 0.102, 0.5),
        ieee80_cs_limit(cs_b1, 2500, 0.5, 1.5, 0.116), "V", rtol=1e-9))

    # Rod resistance, Eq. (55) with n_R = 1 (the docstring cites IEEE 80).
    cases.append(Case(
        "IEEE80-Eq55-rod", IEEE80, "Clause 14.3, Eq. (55) with n_R = 1",
        "Single rod: rho = 400, L = 7.5 m, d = 16 mm (formula identity)",
        "resistance_pipe",
        lambda: e.resistance_pipe(400, 7.5, 0.016),
        ieee80_eq55_single_rod(400, 7.5, 0.016), "ohm", rtol=1e-9))

    return cases


# ---------------------------------------------------------------------------
# Closed-form cases: ENA EREC S34
# ---------------------------------------------------------------------------

# Figure B.1 (p. 48): factor k against the number of rods. Values for 4 and 10
# rods are those used in the worked examples (sections 6.5.1 and 6.1.1); the
# rest were read from the chart (reading accuracy about +/-0.1).
S34_FIG_B1_K = [
    (4, 2.6, "6.5.1"), (10, 4.9, "6.1.1"),
    (16, 5.95, None), (24, 6.8, None), (32, 7.43, None), (40, 7.97, None),
    (48, 8.4, None), (56, 8.7, None), (64, 8.97, None), (68, 9.07, None),
    (72, 9.16, None),
]


def _s34_cases() -> List[Case]:
    cases = []

    # Formula R1, rod electrode.
    cases.append(Case(
        "S34-6.1.1-RR", S34, "6.1.1 (p. 21), Formula R1",
        "Single rod: rho = 75, L = 3.6 m, d = 16 mm", "resistance_pipe",
        lambda: e.resistance_pipe(75, 3.6, 0.016), 21.6, "ohm"))
    cases.append(Case(
        "S34-6.5.1-RR", S34, "6.5.1 (p. 38), Formula R1",
        "Single rod: rho = 50, L = 2.4 m, d = 16 mm", "resistance_pipe",
        lambda: e.resistance_pipe(50, 2.4, 0.016), 20.19, "ohm"))

    # Formula R2, plate electrode. No worked example in S34, so the formula
    # is checked as printed, including its limits: a disc on the surface
    # (rho/4r, exact) and a deeply buried disc (rho/8r, exact).
    for cid, A, h in [("S34-R2-plate-1m2-0.5m", 1.0, 0.5),
                      ("S34-R2-plate-0.84m2-0.6m", 0.915 ** 2, 0.6)]:
        cases.append(Case(
            cid, S34, "B.1.3, Formula R2",
            "Plate: rho = 100, A = {:.3g} m2, h = {} m (formula identity)".format(A, h),
            "resistance_plate",
            (lambda A=A, h=h: e.resistance_plate(100, A, h)),
            s34_r2_plate(100, A, h), "ohm", rtol=1e-9))
    cases.append(Case(
        "S34-R2-plate-surface-limit", S34, "B.1.3, Formula R2 (h -> 0)",
        "Plate at the surface tends to rho/(4r)", "resistance_plate",
        lambda: e.resistance_plate(100, 1.0, 1e-9),
        100 / (4 * sqrt(1 / pi)), "ohm", rtol=1e-6))
    cases.append(Case(
        "S34-R2-plate-deep-limit", S34, "B.1.3, Formula R2 (h -> infinity)",
        "Deep plate tends to rho/(8r)", "resistance_plate",
        lambda: e.resistance_plate(100, 1.0, 1e6),
        100 / (8 * sqrt(1 / pi)), "ohm", rtol=1e-5))

    # Formula R6, combined grid and rods.
    # Appendix B defines b = w/pi (w = tape width). The worked examples
    # substitute b = 0.01 m directly (the conductor diameter), i.e. they
    # correspond to w = pi * 0.01 in the package's parameterisation.
    r6_note = ("Worked example substitutes b = 0.01 m; Appendix B defines "
               "b = w/pi, so w = pi x 0.01 m is passed.")
    cases.append(Case(
        "S34-6.1.1-RE", S34, "6.1.1 (p. 21), Formulae R4-R6",
        "30 m x 20 m grid (L_E = 140 m) + 10 rods x 3.6 m, rho = 75",
        "resistance_grid_with_rods",
        lambda: e.resistance_grid_with_rods(75, 600, 140, 3.6, 10, 10, 0.016, pi * 0.01),
        1.62, "ohm", note=r6_note))
    cases.append(Case(
        "S34-6.5.1-RE", S34, "6.5.1 (p. 38), Formulae R4-R6",
        "3 m x 3 m grid (L_E = 12 m) + 4 rods x 2.4 m, rho = 50",
        "resistance_grid_with_rods",
        lambda: e.resistance_grid_with_rods(50, 9, 12, 2.4, 4, 3, 0.016, pi * 0.01),
        6.57, "ohm", note=r6_note))
    cases.append(Case(
        "S34-6.1.1-RE-w=d", S34, "6.1.1 (p. 21) vs B.1.7",
        "Same as S34-6.1.1-RE but passing w = 0.01 m (b = w/pi)",
        "resistance_grid_with_rods",
        lambda: e.resistance_grid_with_rods(75, 600, 140, 3.6, 10, 10, 0.016, 0.01),
        1.62, "ohm", rtol=None,
        note=("Shows the effect of S34's b inconsistency: Formula R6 as "
              "printed (b = w/pi) gives a lower R_E than the worked example.")))
    cases.append(Case(
        "S34-R6-identity", S34, "B.1.5-B.1.7, Formulae R4, R5, R6",
        "R_E vs Formulae R4-R6 as printed (k = 4.9, b = w/pi)",
        "resistance_grid_with_rods",
        lambda: e.resistance_grid_with_rods(75, 600, 140, 3.6, 10, 10, 0.016, 0.025),
        _s34_r6_reference(75, 600, 140, 3.6, 10, 10, 0.016, 0.025, k=4.9),
        "ohm", rtol=0.002,
        note="0.2 % allows for the package's interpolated k (4.854) vs 4.9."))

    # The same examples with b passed directly, as the worked examples do.
    b_note = "b = 0.01 m (conductor diameter) passed directly, as in the worked example."
    for cid, clause, desc, args, ref in [
        ("S34-6.1.1-RE-b", "6.1.1 (p. 21), Formulae R4-R6",
         "Grid (L_E = 140 m) + 10 rods, b = 0.01 m",
         (75, 600, 140, 3.6, 10, 10, 0.016), 1.62),
        ("S34-6.1.1-RE-rebar-b", "6.1.1 (p. 23), Formulae R4-R6",
         "Grid + re-bar (L_E = 926 m) + 10 rods, b = 0.01 m",
         (75, 600, 926, 3.6, 10, 10, 0.016), 1.43),
        ("S34-6.5.1-RE-b", "6.5.1 (p. 38), Formulae R4-R6",
         "3 m x 3 m grid (L_E = 12 m) + 4 rods, b = 0.01 m",
         (50, 9, 12, 2.4, 4, 3, 0.016), 6.57),
    ]:
        cases.append(Case(
            cid, S34, clause, desc, "resistance_grid_with_rods",
            (lambda args=args: e.resistance_grid_with_rods(*args, None, b=0.01)),
            ref, "ohm", note=b_note))

    # Factor k (Figure B.1), backed out from resistance_grid_with_rods.
    for N, k_ref, src in S34_FIG_B1_K:
        clause = ("Figure B.1 (p. 48); value used in " + src) if src else \
            "Figure B.1 (p. 48), read from chart"
        cases.append(Case(
            "S34-FigB1-k-N{}".format(N), S34, clause,
            "Factor k for {} rods (from resistance_grid_with_rods)".format(N),
            "resistance_grid_with_rods",
            (lambda N=N: s34_k_from_package(100, 10000, 400, 3.6, N, 400 / N, 0.016, 0.025)),
            k_ref, "-", rtol=None, atol=0.15 if src is None else 0.05,
            note=("Chart reading accuracy about +/-0.1." if src is None else "")))
    # Beyond the chart, k must stay bounded at the chart end value rather than
    # be extrapolated (a polynomial fit used to give k = 72.9 at 100 rods).
    cases.append(Case(
        "S34-FigB1-k-N100-beyond-chart", S34,
        "Figure B.1 (p. 48), chart end value (72 rods)",
        "Factor k for 100 rods, outside the 0-72 rod chart range",
        "resistance_grid_with_rods",
        lambda: s34_k_from_package(100, 10000, 400, 3.6, 100, 4.0, 0.016, 0.025),
        9.16, "-", rtol=None, atol=0.15,
        note="Figure B.1 ends at 72 rods; k should be held near its end value."))

    # Formula R7 + R10 / resistance_parallel: section 6.5.6 option 2.
    r_eh = s34_r7_strip(50, 20, 0.00944)
    cases.append(Case(
        "S34-6.5.6-Rpar", S34, "6.5.6 (p. 40), Formulae R7, R10",
        "Substation 6.57 ohm in parallel with two 20 m electrodes", "resistance_parallel",
        lambda: e.resistance_parallel(6.57, r_eh, r_eh), 1.33, "ohm"))

    # resistance_strip methods (formula identities).
    cases.append(Case(
        "S34-R7-strip-method", S34, "B.1.8, Formula R7",
        "resistance_strip(method='S34'): 20 m x 25 mm tape (d = half-width)",
        "resistance_strip",
        lambda: e.resistance_strip(100, 20, 0.5, 0.025, method='S34'),
        s34_r7_strip(100, 20, 0.0125), "ohm", rtol=1e-9))
    cases.append(Case(
        "IEEE142-T4-5-strip-method", IEEE142, "Table 4-5, buried horizontal strip",
        "resistance_strip(method='IEEE142'): 20 m x 25 mm x 3 mm tape at 0.5 m",
        "resistance_strip",
        lambda: e.resistance_strip(100, 20, 0.5, 0.025, method='IEEE142', thickness=0.003),
        ieee142_dwight_strip(100, 20, 0.025, 0.003, 0.5), "ohm", rtol=1e-9))

    # Appendix D.4.1, C-factor method.
    cases.append(Case(
        "S34-6.2.3-IE%", S34, "6.2.3 Table 4 (p. 28), Formula D.4.1",
        "Ground return ratio: 33 kV, 3 km 185 mm2 triplex, C = 77", "current_ratio",
        lambda: e.current_ratio(75, 1.68, 77, 33, 185, 3), 0.168, "-"))
    cases.append(Case(
        "S34-6.2.3-IE", S34, "6.2.3 Table 4 (p. 28), Formula D.4.1",
        "Ground return current for IF = 1896 A", "current_ratio",
        lambda: 1896 * e.current_ratio(75, 1.68, 77, 33, 185, 3), 318, "A"))
    cases.append(Case(
        "S34-6.5.2-IE%", S34, "6.5.2 Table 13 (p. 39), Formula D.4.1",
        "Ground return ratio: 11 kV, 1 km 185 mm2 triplex, C = 47", "current_ratio",
        lambda: e.current_ratio(50, 6.67, 47, 11, 185, 1), 0.0241, "-"))
    cases.append(Case(
        "S34-6.5.2-IE", S34, "6.5.2 Table 13 (p. 39), Formula D.4.1",
        "Ground return current for IF = 3000 A", "current_ratio",
        lambda: 3000 * e.current_ratio(50, 6.67, 47, 11, 185, 1), 72.3, "A"))

    # Formula C2, limiting surface current density.
    cases.append(Case(
        "S34-6.5.5-Jlimit", S34, "6.5.5 (p. 40), Formula C2",
        "Limiting current density, rho = 50, t = 3 s", "max_current_density",
        lambda: e.max_current_density(50, 3), 620, "A/m2",
        note="Published as 0.62e-3 A/mm2 (2 significant figures)."))
    cases.append(Case(
        "S34-6.5.5-Jlimit-S34", S34, "6.5.5 (p. 40), Formula C2",
        "Limiting current density, rho = 50, t = 3 s, method='S34'",
        "max_current_density",
        lambda: e.max_current_density(50, 3, method='S34'), 620, "A/m2",
        note="Published as 0.62e-3 A/mm2 (2 significant figures)."))
    cases.append(Case(
        "S34-C2-coefficient-S34", S34, "B.2.2, Formula C2",
        "Coefficient with method='S34' (formula identity)", "max_current_density",
        lambda: e.max_current_density(1, 1, method='S34'), 1e3 * sqrt(57.7),
        "A/m2", rtol=1e-9))
    cases.append(Case(
        "S34-C2-coefficient", S34, "B.2.2, Formula C2",
        "Coefficient: J * sqrt(rho * t)", "max_current_density",
        lambda: e.max_current_density(1, 1), 1e3 * sqrt(57.7), "A/m2", rtol=None,
        note=("Package uses 7.57e3 (IS 3043); Formula C2 gives "
              "1e3 x sqrt(57.7) = 7.596e3. The package is 0.34 % lower "
              "(conservative).")))

    # Surface potential formulae, Appendix B.3, against worked examples.
    # Section 6.1 (Substation B): 30 m x 20 m grid with 10 rods, h = 0.6 m,
    # d = 0.01 m, rho = 75, IE = 1477 A, L_T = 176 m, L_p = 136 m,
    # D = 15 m, nA = 2, nB = 4.
    p1 = (75, 1477, 176, 136, 0.6, 0.01, 15, 2, 4)
    cases.append(Case(
        "S34-6.1.4-P1", S34, "6.1.4 (p. 25), Formula P1",
        "External touch potential 1 m from grid edge, Substation B",
        "e_touch_external", lambda: e.e_touch_external(*p1), 648, "V"))
    cases.append(Case(
        "S34-6.1.4-P1-cross-member", S34, "6.1.4 (p. 25), Formula P1",
        "As S34-6.1.4-P1 with a central cross member (L_T = 206 m, D = 10 m, nA = 3)",
        "e_touch_external",
        lambda: e.e_touch_external(75, 1477, 206, 136, 0.6, 0.01, 10, 3, 4),
        602, "V"))
    cases.append(Case(
        "S34-6.1.5-P2", S34, "6.1.5 (p. 26), Formula P2",
        "Touch potential at a separately earthed fence 2 m from the grid",
        "e_touch_fence", lambda: e.e_touch_fence(*p1), 169, "V",
        note=("Reproduced with L_T = 176 m as divisor. Formula P2 as printed "
              "divides by L_p, which gives 218 V.")))
    cases.append(Case(
        "S34-6.1.6-P4", S34, "6.1.6 (p. 26), Formula P4",
        "Internal touch potential (IEEE 80 mesh voltage), grid + 10 rods",
        "e_mesh_step_grid",
        lambda: e.e_mesh_step_grid(75, 30, 20, 3.6, 2, 4, 10, 0.01, 0.6, 1477)[0],
        657, "V", rtol=None,
        note=("S34 does not state D for P4. With D = 15 m (as in P1) the "
              "package gives 689 V (+4.9 %); D = 12.5 m reproduces 657 V.")))
    cases.append(Case(
        "S34-6.1.7-P7", S34, "6.1.7 (p. 26), Formula P7",
        "Distance to the 840 V contour, A = 600 m2, EPR = 2393 V",
        "contour_distance", lambda: e.contour_distance(600, 840, 2393),
        12.5, "m"))
    # Appendix I.5: 3.6 m rod, rho = 75, I = 200 A.
    for x, ref in ((9, 259), (50, 48)):
        cases.append(Case(
            "S34-I.5-P6.2-x{}".format(x), S34, "I.5 (p. 78), Formula P6.2",
            "Surface potential {} m from a 3.6 m rod, I = 200 A".format(x),
            "surface_potential_rod",
            (lambda x=x: e.surface_potential_rod(75, 200, 3.6, x)), ref, "V",
            atol=0.5,
            note="Published to the nearest volt."))
    # Formulae with no worked example: identities and consistency checks.
    cases.append(Case(
        "S34-P6.3-identity", S34, "B.3.7, Formula P6.3",
        "Grid surface potential, R_E = 1.62, I = 1477 A, x = 30 m (formula identity)",
        "surface_potential_grid",
        lambda: e.surface_potential_grid(75, 1477, 1.62, 30),
        s34_p6_3_grid(75, 1477, 1.62, 30), "V", rtol=1e-9))
    cases.append(Case(
        "S34-P6.3-at-r-equals-EPR", S34, "B.3.7, Formula P6.3 (x = r)",
        "Grid surface potential at x = r = rho/(4 R_E) equals the EPR",
        "surface_potential_grid",
        lambda: e.surface_potential_grid(75, 1477, 1.62, 75 / (4 * 1.62)),
        1477 * 1.62, "V", rtol=1e-9))
    cases.append(Case(
        "S34-P6.1-identity", S34, "B.3.7, Formula P6.1",
        "Hemisphere surface potential, rho = 75, I = 200 A, x = 9 m",
        "surface_potential_hemisphere",
        lambda: e.surface_potential_hemisphere(75, 200, 9),
        75 * 200 / (2 * pi * 9), "V", rtol=1e-9))
    cases.append(Case(
        "S34-P5-vs-P6.3", S34, "B.3.6, Formula P5 vs P6.3",
        "Step potential at x = 20 m equals V(20) - V(21) from P6.3",
        "e_step_external",
        lambda: e.e_step_external(75, 1477, 1.62, 20),
        s34_p6_3_grid(75, 1477, 1.62, 20) - s34_p6_3_grid(75, 1477, 1.62, 21),
        "V", rtol=1e-9))
    k_fe = 0.5 * log(0.6 / 0.01) - 0.25 * log((5 ** 2 + 0.5 ** 2) ** 2) + 0.25 * log(5 ** 4 + 5 ** 2)
    cases.append(Case(
        "S34-P3-identity", S34, "B.3.4, Formula P3",
        "Fence touch potential with peripheral conductor, S = 5 m (formula identity)",
        "e_touch_fence_peripheral",
        lambda: e.e_touch_fence_peripheral(75, 1477, 176, 136, 0.6, 0.01, 5),
        k_fe * (0.7 + 0.3 * 176 / 136) * 75 * 1477 / 176, "V", rtol=1e-9,
        note=("No worked example in S34. k_fe has no 1/pi factor as printed, "
              "unlike k_e; the result is not checked against a published value.")))

    return cases


def _s34_r6_reference(rho, A, L_E, L_R, N, S, d, w, k):
    R1 = s34_r4_grid(rho, A, L_E)
    R_R = s34_r1_rod(rho, L_R, d)
    alpha = rho / (2 * pi * R_R * S)
    R2 = R_R * (1 + k * alpha) / N
    R12 = R1 - rho / (pi * L_E) * (log(L_R / (w / pi)) - 1)
    return (R1 * R2 - R12 ** 2) / (R1 + R2 - 2 * R12)


# ---------------------------------------------------------------------------
# Closed-form cases: ENA TS 41-24
# ---------------------------------------------------------------------------

# Table 8 (p. 37): maximum current rating of typical electrodes, calculated
# with Formula C2 of S34. Columns: (label, surface area in m^2, unit).
TS4124_T8_ELECTRODES = [
    ("rod16", pi * 0.016 * 1.0, "A/m"),        # 16 mm rod, per metre
    ("plate915", 2 * 0.915 ** 2, "A"),          # both faces
    ("plate1220", 2 * 1.22 ** 2, "A"),          # both faces
    ("tape25x4", 2 * (0.025 + 0.004), "A/m"),   # perimeter, per metre
]
TS4124_T8 = {
    # rho: (3 s: rod, plate915, plate1220, tape), (1 s: rod, plate915, plate1220, tape)
    10: ((69.7, 2322, 4128, 80.4), (120.7, 4022, 7151, 139.3)),
    30: ((40.2, 1341, 2384, 46.4), (69.7, 2322, 4128, 80.4)),
    40: ((34.9, 1161, 2064, 40.2), (60.4, 2011, 3575, 69.7)),
    50: ((31.2, 1039, 1846, 36), (54, 1799, 3198, 62.3)),
    60: ((28.5, 948, 1685, 32.8), (49.3, 1642, 2919, 56.9)),
    70: ((26.3, 878, 1560, 30.4), (45.6, 1520, 2703, 52.7)),
    80: ((24.6, 821, 1460, 28.4), (42.7, 1422, 2528, 49.3)),
    100: ((22, 734, 1306, 25.4), (38.2, 1272, 2261, 44.1)),
    150: ((18, 600, 1066, 20.8), (31.2, 1038, 1846, 36)),
    200: ((15.6, 519, 923, 18), (27, 899, 1599, 31.2)),
    250: ((13.9, 464, 826, 16.1), (24.1, 804, 1430, 27.9)),
    300: ((12.7, 424, 754, 14.7), (22, 734, 1306, 25.4)),
}


def _ts4124_cases() -> List[Case]:
    cases = []
    # Tables 1 and 2 (p. 20-21), checked where S34 quotes the same limits.
    for cid, clause, desc, f, ref in [
        ("TS41-24-T1-0.4s-chippings75", "Table 1 (p. 20); quoted in S34 6.1.4",
         "Touch limit, 0.4 s, shoes on 75 mm chippings",
         lambda: e.touch_limit_ts4124(0.4, 'shoes_chippings_75'), 944),
        ("TS41-24-T1-1s-soil", "Table 1 (p. 20); quoted in S34 I.5",
         "Touch limit, 1 s, shoes on soil",
         lambda: e.touch_limit_ts4124(1, 'shoes_soil'), 233),
        ("TS41-24-T1-0.6s-soil", "Table 1 (p. 20); S34 6.1.7 uses 2 x UTP = 840 V",
         "Touch limit, 0.6 s, shoes on soil",
         lambda: e.touch_limit_ts4124(0.6, 'shoes_soil'), 420),
        ("TS41-24-T1-10s-hand", "Table 1 (p. 20)",
         "Touch limit, >= 10 s, hand-to-hand (value held beyond 10 s)",
         lambda: e.touch_limit_ts4124(30, 'hand_to_hand'), 114),
        ("TS41-24-T1-0.45s-conservative", "Table 1 (p. 20)",
         "Touch limit at 0.45 s, 75 mm chippings, conservative (0.5 s value)",
         lambda: e.touch_limit_ts4124(0.45, 'shoes_chippings_75'), 650),
        ("TS41-24-T1-0.45s-linear", "Table 1 (p. 20)",
         "Touch limit at 0.45 s, 75 mm chippings, linear interpolation",
         lambda: e.touch_limit_ts4124(0.45, 'shoes_chippings_75', 'linear'),
         (944 + 650) / 2),
        ("TS41-24-T2-1s-bare", "Table 2 (p. 21)",
         "Step limit, 1 s, bare feet",
         lambda: e.step_limit_ts4124(1, 'bare_feet'), 2249),
        ("TS41-24-T2-3s-chippings150", "Table 2 (p. 21)",
         "Step limit, 3 s, shoes on 150 mm chippings",
         lambda: e.step_limit_ts4124(3, 'shoes_chippings_150'), 16074),
    ]:
        cases.append(Case(cid, TS4124, clause, desc,
                          "touch_limit_ts4124" if "-T1-" in cid else "step_limit_ts4124",
                          f, ref, "V", rtol=1e-9))
    for rho, by_time in TS4124_T8.items():
        for t, row in zip((3, 1), by_time):
            for (label, area, unit), ref in zip(TS4124_T8_ELECTRODES, row):
                cases.append(Case(
                    "TS41-24-T8-{}s-{}-rho{}".format(t, label, rho), TS4124,
                    "5.5.2 Table 8 (p. 37)",
                    "Electrode rating {}, rho = {}, t = {} s".format(label, rho, t),
                    "max_current_density",
                    (lambda rho=rho, t=t, area=area: e.max_current_density(rho, t) * area),
                    ref, unit,
                    note="Rating = J x surface area; table values have 3-4 significant figures."))
    return cases


# ---------------------------------------------------------------------------
# Numerical solver (Network) cases
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _rod_network(delta, fast=True):
    n = e.Network(75, 1000)
    n.add_rod([0, 0, 0], 0.008, 3.6)
    if fast:
        n.generate_model_fast(delta)
    else:
        n.generate_model(delta)
    n.solve_model()
    return n


@functools.lru_cache(maxsize=None)
def _rod_network_i200():
    n = e.Network(75, 200)
    n.add_rod([0, 0, 0], 0.008, 3.6)
    n.generate_model_fast(DELTA_ROD_FINE)
    n.solve_model()
    return n


@functools.lru_cache(maxsize=None)
def _two_subnet_network():
    n = e.Network(75, 1000)
    n.add_rod([0, 0, 0], 0.008, 3.6)
    n.add_subnet(0)
    n.add_rod([20, 0, 0], 0.008, 3.6)
    n.generate_model_fast(DELTA_ROD_FINE)
    n.solve_model()
    return n


@functools.lru_cache(maxsize=None)
def _plate_network(delta):
    n = e.Network(100, 1000)
    n.add_plate([0, 0, -0.5], 1, 1, n_cap=[0, 0, 1], h_cap=[1, 0, 0])
    n.generate_model_fast(delta)
    n.solve_model()
    return n


@functools.lru_cache(maxsize=None)
def _strip_network(delta):
    n = e.Network(100, 1000)
    n.add_strip([0, 0, -0.5], [20, 0, -0.5], 0.025)
    n.generate_model_fast(delta)
    n.solve_model()
    return n


# IEEE 80 Annex B grid conductor diameter (m). The solver only has flat strips,
# so the round conductor is represented by a strip of width 2d: a thin strip of
# width w has an equivalent radius of w/4. The alternative "equal surface
# area" width pi*d/2 (the convention NetworkElementPipe uses) is reported as an
# informational case because the mesh voltage is sensitive to this choice.
IEEE_D = 0.01
IEEE_STRIP_W = 2 * IEEE_D
IEEE_STRIP_W_AREA = pi * IEEE_D / 2
# Rod diameter is not stated in Annex B; 16 mm is assumed.
IEEE_ROD_RADIUS = 0.008


@functools.lru_cache(maxsize=None)
def _ieee_grid_network(with_rods, strip_w=IEEE_STRIP_W, delta=DELTA):
    """IEEE 80 Annex B Example 1 / 2 geometry: 70 m x 70 m, 11 x 11
    conductors at 0.5 m depth; optionally 20 rods of 7.5 m on a 14 m pitch
    around the perimeter as in Figure B.2."""
    n = e.Network(400, 1908)
    n.add_mesh([0, 0, -0.5], 70, 70, 11, 11, strip_w)
    if with_rods:
        for i in range(6):
            n.add_rod([i * 14, 0, -0.5], IEEE_ROD_RADIUS, 7.5)
            n.add_rod([i * 14, 70, -0.5], IEEE_ROD_RADIUS, 7.5)
        for i in range(1, 5):
            n.add_rod([0, i * 14, -0.5], IEEE_ROD_RADIUS, 7.5)
            n.add_rod([70, i * 14, -0.5], IEEE_ROD_RADIUS, 7.5)
    n.generate_model_fast(delta)
    n.solve_model()
    return n


# IEEE 80 Example 4 (Figure B.4): L-shaped grid on a 7 m mesh. Vertical arm
# x = 0-35 m, y = 0-105 m; horizontal arm x = 35-70 m, y = 0-35 m.
IEEE_B4_RODS = ([(x, 0) for x in (0, 14, 28, 42, 56, 70)] + [(70, 14)]
                + [(x, 35) for x in (42, 56, 70)]
                + [(35, y) for y in (42, 56, 70, 84, 98)]
                + [(x, 105) for x in (0, 14, 28)]
                + [(0, y) for y in (14, 28, 42, 56, 70, 84)])
IEEE_B4_CORNERS = ((0, 0), (70, 0), (70, 35), (35, 105), (0, 105))


@functools.lru_cache(maxsize=None)
def _ieee_l_grid_network(delta=DELTA):
    z = -0.5
    n = e.Network(400, 1908)
    for x in range(0, 36, 7):
        n.add_strip([x, 0, z], [x, 105, z], IEEE_STRIP_W)
    for x in range(42, 71, 7):
        n.add_strip([x, 0, z], [x, 35, z], IEEE_STRIP_W)
    for y in range(0, 36, 7):
        n.add_strip([0, y, z], [70, y, z], IEEE_STRIP_W)
    for y in range(42, 106, 7):
        n.add_strip([0, y, z], [35, y, z], IEEE_STRIP_W)
    for x, y in IEEE_B4_RODS:
        n.add_rod([x, y, z], IEEE_ROD_RADIUS, 7.5)
    n.generate_model_fast(delta)
    n.solve_model()
    return n


def _ieee_l_mesh_voltage():
    n = _ieee_l_grid_network()
    meshes = ((0, 0), (63, 0), (63, 28), (28, 98), (0, 98))
    return max(float(n.mesh_voltage(e.rectangle(c, 7, 7))[1]) for c in meshes)


def _ieee_l_step_voltage():
    n = _ieee_l_grid_network()
    return max(float(n.step_voltage(e.rectangle((x - 1, y - 1), 2, 2), mesh_no=20)[1])
               for x, y in IEEE_B4_CORNERS)


@functools.lru_cache(maxsize=None)
def _ieee_mesh_voltage(with_rods, strip_w=IEEE_STRIP_W):
    n = _ieee_grid_network(with_rods, strip_w)
    return float(n.mesh_voltage(e.rectangle((0, 0), 7, 7))[1])


@functools.lru_cache(maxsize=None)
def _ieee_step_voltage(with_rods):
    n = _ieee_grid_network(with_rods)
    return float(n.step_voltage(e.rectangle((-1, -1), 2, 2), mesh_no=20)[1])


def _numerical_cases() -> List[Case]:
    cases = []
    delta_note = "Discretisation {} m.".format(DELTA)
    epri = ("Reference is the EPRI TR-100622 computer result quoted in "
            "IEEE 80 Annex B for comparison. Strip width 2d = {} m. ".format(IEEE_STRIP_W))

    cases.append(Case(
        "NUM-rod-vs-S34-R1", S34, "B.1.2 Formula R1 / 6.1.1",
        "Network: single rod 3.6 m, 16 mm, rho = 75", "Network.get_resistance",
        lambda: _rod_network(DELTA_ROD).get_resistance()[0],
        s34_r1_rod(75, 3.6, 0.016), "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL,
        note="Discretisation {} m.".format(DELTA_ROD)))
    cases.append(Case(
        "NUM-plate-vs-S34-R2", S34, "B.1.3 Formula R2",
        "Network: horizontal 1 m x 1 m plate at 0.5 m, rho = 100",
        "Network.get_resistance",
        lambda: _plate_network(DELTA).get_resistance()[0],
        s34_r2_plate(100, 1.0, 0.5), "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL, note=delta_note))

    cases.append(Case(
        "NUM-IEEE80-B1-Rg", IEEE80, "Annex B.1 Step 9 (EPRI comparison)",
        "Network: 70 m x 70 m grid, no rods", "Network.get_resistance",
        lambda: _ieee_grid_network(False).get_resistance()[0], 2.67, "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL, note=epri + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B1-Em", IEEE80, "Annex B.1 Step 9 (EPRI comparison)",
        "Network: corner-mesh touch voltage, no rods", "Network.mesh_voltage",
        lambda: _ieee_mesh_voltage(False), 984.3, "V",
        rtol=RTOL_NUM_VOLTAGE, kind=NUMERICAL, note=epri + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B2-Rg", IEEE80, "Annex B.2 Step 12 (EPRI comparison)",
        "Network: 70 m x 70 m grid, 20 rods x 7.5 m", "Network.get_resistance",
        lambda: _ieee_grid_network(True).get_resistance()[0], 2.52, "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL, note=epri + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B2-Em", IEEE80, "Annex B.2 Step 12 (EPRI comparison)",
        "Network: corner-mesh touch voltage, 20 rods", "Network.mesh_voltage",
        lambda: _ieee_mesh_voltage(True), 756.2, "V",
        rtol=RTOL_NUM_VOLTAGE, kind=NUMERICAL, note=epri + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B2-Es", IEEE80, "Annex B.2 Step 12 (EPRI comparison)",
        "Network: step voltage at grid corner, 20 rods", "Network.step_voltage",
        lambda: _ieee_step_voltage(True), 459.1, "V",
        rtol=RTOL_NUM_VOLTAGE, kind=NUMERICAL, note=epri + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B1-Em-w=pi.d/2", IEEE80, "Annex B.1 Step 9 (EPRI comparison)",
        "Network: as NUM-IEEE80-B1-Em with strip width pi*d/2 (equal area)",
        "Network.mesh_voltage",
        lambda: _ieee_mesh_voltage(False, IEEE_STRIP_W_AREA), 984.3, "V",
        rtol=None, kind=NUMERICAL,
        note=("Sensitivity to the round-conductor-to-strip equivalence; "
              "compare with NUM-IEEE80-B1-Em.")))

    # IEEE 80 Example 4, L-shaped grid with 24 perimeter rods (Figure B.4).
    epri4 = ("Reference is the EPRI TR-100622 result quoted in IEEE 80 Annex B.4. "
             "Rod positions read from Figure B.4. ")
    cases.append(Case(
        "NUM-IEEE80-B4-Rg", IEEE80, "Annex B.4 Step 12 (EPRI comparison)",
        "Network: L-shaped grid, 24 rods x 7.5 m", "Network.get_resistance",
        lambda: _ieee_l_grid_network().get_resistance()[0], 2.34, "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL, note=epri4 + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B4-Em", IEEE80, "Annex B.4 Step 12 (EPRI comparison)",
        "Network: largest corner-mesh touch voltage, L-shaped grid",
        "Network.mesh_voltage", _ieee_l_mesh_voltage, 742.9, "V",
        rtol=RTOL_NUM_VOLTAGE, kind=NUMERICAL,
        note=epri4 + "Maximum over the five outer corner meshes. " + delta_note))
    cases.append(Case(
        "NUM-IEEE80-B4-Es", IEEE80, "Annex B.4 Step 12 (EPRI comparison)",
        "Network: largest step voltage at the outer corners, L-shaped grid",
        "Network.step_voltage", _ieee_l_step_voltage, 441.8, "V",
        rtol=RTOL_NUM_VOLTAGE, kind=NUMERICAL,
        note=epri4 + "Maximum over the five outer corners. " + delta_note))

    # generate_model (non-fast path) must agree with generate_model_fast.
    cases.append(Case(
        "NUM-generate_model-rod", S34, "B.1.2 Formula R1 / 6.1.1",
        "Network.generate_model: single rod 3.6 m, 16 mm, rho = 75",
        "Network.generate_model",
        lambda: _rod_network(DELTA_ROD, fast=False).get_resistance()[0],
        s34_r1_rod(75, 3.6, 0.016), "ohm",
        rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL,
        note="Discretisation {} m.".format(DELTA_ROD)))
    cases.append(Case(
        "NUM-generate_model-vs-fast", S34, "-",
        "Network.generate_model / generate_model_fast resistance ratio (single rod)",
        "Network.generate_model",
        lambda: (_rod_network(DELTA_ROD, fast=False).get_resistance()[0]
                 / _rod_network(DELTA_ROD).get_resistance()[0]),
        1.0, "-",
        rtol=1e-3, kind=NUMERICAL,
        note="Both paths use the same disc elements; results are rounded to 1 mohm."))

    # get_point_potential against S34 Formula P6.2 (surface potential near a
    # rod, uniform current density). 3.6 m rod, rho = 75, I = 200 A.
    for x, tol in ((1, RTOL_NUM_VOLTAGE), (9, RTOL_NUM_RESISTANCE), (50, RTOL_NUM_RESISTANCE)):
        cases.append(Case(
            "NUM-point-potential-x{}".format(x), S34, "B.3.7 Formula P6.2",
            "Network.get_point_potential {} m from a 3.6 m rod, I = 200 A".format(x),
            "Network.get_point_potential",
            (lambda x=x: _rod_network_i200().get_point_potential([x, 0, 0])),
            e.surface_potential_rod(75, 200, 3.6, x), "V",
            rtol=tol, kind=NUMERICAL,
            note=("P6.2 assumes uniform current density along the rod, so it "
                  "is least accurate close to the rod. Discretisation {} m."
                  ).format(DELTA_ROD_FINE)))

    # Two sub-networks: two 3.6 m rods 20 m apart, current injected into the
    # first only. The second rod's potential per ampere is the mutual
    # resistance, IEEE 142 Table 4-5 (two rods, s > L).
    s_sub = 20.0
    r12 = 75 / (2 * pi * s_sub) * (1 - 3.6 ** 2 / (3 * s_sub ** 2) + 2 * 3.6 ** 4 / (5 * s_sub ** 4))
    cases.append(Case(
        "NUM-subnet-mutual", IEEE142, "Table 4-5, two ground rods, s > L",
        "Network.add_subnet: potential of an unenergised rod 20 m away, per ampere",
        "Network.add_subnet", lambda: _two_subnet_network().gpr()[1] / 1000,
        r12, "ohm", rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL,
        note="Mutual term of the two-rod formula. Discretisation {} m.".format(DELTA_ROD_FINE)))
    cases.append(Case(
        "NUM-subnet-self", S34, "B.1.2 Formula R1",
        "Network.add_subnet: energised rod resistance with a second sub-network present",
        "Network.add_subnet", lambda: _two_subnet_network().gpr()[0] / 1000,
        s34_r1_rod(75, 3.6, 0.016), "ohm", rtol=RTOL_NUM_RESISTANCE, kind=NUMERICAL,
        note=("The unenergised rod carries no net current, so the energised "
              "rod should match Formula R1. Discretisation {} m.").format(DELTA_ROD_FINE)))

    # Discretisation sensitivity (informational).
    for delta in (0.4, 0.2, 0.1, 0.05, 0.025):
        cases.append(Case(
            "NUM-rod-sensitivity-d{}".format(delta), S34, "B.1.2 Formula R1",
            "Network: single rod, discretisation {} m".format(delta),
            "Network.get_resistance",
            (lambda delta=delta: _rod_network(delta).get_resistance()[0]),
            s34_r1_rod(75, 3.6, 0.016), "ohm", rtol=None, kind=NUMERICAL))

    # resistance_strip cites IS 3043, which is not among the reference
    # standards. Compared with S34 Formula R7 and with the solver for context.
    cases.append(Case(
        "INFO-strip-vs-S34-R7", S34, "B.1.8 Formula R7",
        "resistance_strip (IS 3043): 20 m x 25 mm tape at 0.5 m, rho = 100",
        "resistance_strip",
        lambda: e.resistance_strip(100, 20, 0.5, 0.025),
        s34_r7_strip(100, 20, 0.0125), "ohm", rtol=None,
        note=("IS 3043 is not in the reference set. R7 is a different "
              "(simplified) formula, so no pass/fail is applied.")))
    cases.append(Case(
        "INFO-strip-vs-Network", S34, "-",
        "Network: same 20 m tape, compared with resistance_strip",
        "Network.get_resistance",
        lambda: _strip_network(DELTA).get_resistance()[0],
        e.resistance_strip(100, 20, 0.5, 0.025), "ohm", rtol=None, kind=NUMERICAL,
        note="Reference here is resistance_strip (IS 3043), not a standard value."))

    return cases


def all_cases() -> List[Case]:
    return _ieee80_cases() + _s34_cases() + _ts4124_cases() + _numerical_cases()
