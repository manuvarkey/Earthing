#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discretisation error sweeps for ``Network.generate_model_fast``.

Regenerates the tables in validation/README.md, "Discretisation guidance":
single rods, strips and plates compared with closed-form references.

    python3 validation/discretisation_sweep.py
    python3 validation/discretisation_sweep.py --only rod

Each table cell is the deviation (%) of the solver resistance from the
reference formula. Runs in under a minute.
"""

import argparse
import os
import sys
import warnings
from math import log, pi, sqrt

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import earthing as e  # noqa: E402


def s34_r1_rod(rho, L, d):
    """ENA EREC S34 Formula R1."""
    return rho / (2 * pi * L) * (log(8 * L / d) - 1)


def s34_r2_plate(rho, A, h):
    """ENA EREC S34 Formula R2."""
    r = sqrt(A / pi)
    return rho / (8 * r) * (1 + r / (2.5 * h + r))


def dwight_wire(rho, length, a, depth):
    """IEEE Std 142-2007 Table 4-5, buried horizontal wire (length 2L)."""
    L = length / 2
    s = 2 * depth
    return rho / (4 * pi * L) * (log(4 * L / a) + log(4 * L / s) - 2 + s / (2 * L)
                                 - s ** 2 / (16 * L ** 2) + s ** 4 / (512 * L ** 4))


def solve(build, delta):
    n = e.Network(100, 1000)
    build(n)
    n.generate_model_fast(delta)
    n.solve_model()
    return n.get_resistance()[0]


def table(title, col_label, cols, rows):
    print("\n" + title)
    print("{:<22}".format(col_label) + "".join("{:>9}".format(c) for c in cols))
    for label, values in rows:
        print("{:<22}".format(label) + "".join(
            "{:>+9.1f}".format(v) if v is not None else "{:>9}".format("-") for v in values))


def sweep_rod():
    ratios = (1, 2, 3, 4, 6, 8, 12)
    rows = []
    for L in (1.2, 2.4, 3.6, 7.5):
        for d in (0.012, 0.016, 0.025):
            ref = s34_r1_rod(100, L, d)
            vals = [(solve(lambda n: n.add_rod([0, 0, 0], d / 2, L), k * d) / ref - 1) * 100
                    for k in ratios]
            rows.append(("L={} m d={} mm".format(L, d * 1000), vals))
    table("Rod vs S34 R1, deviation (%)", "step / diameter", ratios, rows)


def sweep_strip():
    ratios = (1, 2, 3, 4, 6, 8, 16)
    rows = []
    for L, w in ((10, 0.025), (20, 0.025), (20, 0.05), (50, 0.025)):
        ref = dwight_wire(100, L, w / 4, 0.5)
        vals = [(solve(lambda n: n.add_strip([0, 0, -0.5], [L, 0, -0.5], w), k * w) / ref - 1) * 100
                for k in ratios]
        rows.append(("L={} m w={} mm".format(L, w * 1000), vals))
    table("Strip at 0.5 m vs Dwight wire (a = w/4), deviation (%)", "step / width", ratios, rows)


def sweep_plate():
    deltas = (0.4, 0.2, 0.1, 0.05, 0.025)
    rows = []
    for side, h in ((0.6, 0.5), (1.0, 0.5), (1.2, 1.0)):
        ref = s34_r2_plate(100, side ** 2, h)
        vals = []
        for delta in deltas:
            if delta > side / 2:
                vals.append(None)
                continue
            r = solve(lambda n: n.add_plate([0, 0, -h], side, side,
                                            n_cap=[0, 0, 1], h_cap=[1, 0, 0]), delta)
            vals.append((r / ref - 1) * 100)
        rows.append(("side={} m h={} m".format(side, h), vals))
    table("Horizontal plate vs S34 R2, deviation (%)", "step (m)", deltas, rows)


SWEEPS = {"rod": sweep_rod, "strip": sweep_strip, "plate": sweep_plate}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=sorted(SWEEPS), help="run one sweep")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # arctan(inf) self term
        for name, sweep in SWEEPS.items():
            if args.only in (None, name):
                sweep()


if __name__ == "__main__":
    main()
