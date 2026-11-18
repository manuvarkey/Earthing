#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  
#  Copyright 2022 Manu Varkey <manuvarkey@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

from math import sin, cos, tan, atan, asin, log, sqrt, pi, inf
import warnings

import numpy as np
from numpy.linalg import norm, solve

import matplotlib.pyplot as plt
from matplotlib import path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from cycler import cycler

__all__ = ["Network",
           "resistance_plate", "resistance_pipe", "resistance_strip",
           "resistance_grid", "resistance_grid_with_rods",
           "e_step_50", "e_touch_50", "e_step_70", "e_touch_70",
           "e_mesh_step", "e_mesh_step_grid",
           "e_touch_external", "e_touch_fence", "e_touch_fence_peripheral",
           "e_step_external", "surface_potential_hemisphere",
           "surface_potential_rod", "surface_potential_grid",
           "contour_distance", "touch_limit_ts4124", "step_limit_ts4124",
           "fault_current", "max_current_density", "current_ratio",
           "resistance_parallel", "rectangle"]

# Limit visibility of modules to __all__
def __dir__():
    return __all__
    
# Global variables
def plot_cycler():
    return (cycler(color=list('grcy')))
eplot_linewidth = 3

## Resistance functions
    
def resistance_plate(rho, A, h, method='IEEE142'):
    """ Resistance of a buried horizontal plate

        rho: resistivity (ohm-m)
        A: plate area, one face (m^2); a square plate is treated as a
           round plate of equal area, radius a = sqrt(A/pi)
        h: burial depth (m)
        method: 'IEEE142' - Dwight, IEEE Std 142-2007 Table 4-5, buried
                            horizontal round plate at depth s/2 (default):
                            rho/(8a) + rho/(4 pi s)(1 - 7a^2/(12 s^2)
                            + 33 a^4/(40 s^4)), s = 2h. Series form; use
                            for h greater than about a/2.
                'S34'     - ENA EREC S34 Formula R2:
                            rho/(8a)(1 + a/(2.5 h + a))
    """
    a = sqrt(A/pi)
    if method == 'IEEE142':
        s = 2*h
        return rho/(8*a) + rho/(4*pi*s)*(1 - 7*a**2/(12*s**2)
                                         + 33*a**4/(40*s**4))
    elif method == 'S34':
        return rho/(8*a)*(1 + a/(2.5*h + a))
    raise ValueError("method must be 'IEEE142' or 'S34'")

def resistance_pipe(rho, L, d):
    """ As per IEEE 80
    
        rho: resistivity (ohm-m)
        L: length of pipe (m)
        d: dia of pipe (m)
    """
    return rho/(2*pi*L)*(log(8*L/d) - 1)

def resistance_strip(rho, L, w, t, method='IEEE142', thickness=0.0):
    """ Resistance of a straight horizontal strip or round conductor

        rho: resistivity (ohm-m)
        L: length of strip (m)
        w: depth of burial (m)
        t: width of strip, or 2 x diameter for a round conductor (m)
        method: formula to use
            'IEEE142' - Dwight, IEEE Std 142-2007 Table 4-5, buried
                        horizontal strip of width t and given thickness
                        (thickness < t/8), at depth w (default). For a
                        round conductor, t = 2 x diameter gives a thin
                        strip of the same equivalent radius (within about
                        1 % of the Table 4-5 buried wire formula).
            'S34'     - ENA EREC S34 Formula R7 (from BS EN 50522; same as
                        IS 3043:2018 clause 14.2.3):
                        rho/(2 pi L) ln(2 L/d), d = t/2 (half-width of tape
                        or diameter of round conductor). Ignores w; valid
                        for burial depth 0.5-0.6 m and L <= 60 sqrt(rho).
        thickness: strip thickness (m), used by 'IEEE142' only
    """
    if method == 'S34':
        return rho/(2*pi*L)*log(2*L/(t/2))
    elif method == 'IEEE142':
        Lh = L/2  # Table 4-5 uses length 2L
        a, b, s = t, thickness, 2*w
        return rho/(4*pi*Lh)*(log(4*Lh/a) + (a**2 - pi*a*b)/(2*(a + b)**2)
                              + log(4*Lh/s) - 1 + s/(2*Lh)
                              - s**2/(16*Lh**2) + s**4/(512*Lh**4))
    raise ValueError("method must be 'IEEE142' or 'S34'")
        
def resistance_grid(rho, A, L, h=0.5):
    """ Return grid resistance as per IEEE 80
    
        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L: total buried length of conductors (m)
        h: depth of grid (m)
    """
    return rho*(1/L + 1/sqrt(20*A)*(1 + 1/(1+h*sqrt(20/A))))

# Factor k against total number of rods, digitised from ENA EREC S34
# Figure B.1 (chart range 0-72 rods)
_S34_FIG_B1_K = np.array([[0, 0], [4, 2.6], [8, 4.3], [12, 5.3], [16, 6],
                          [20, 6.5], [24, 6.8], [40, 8], [64, 9], [72, 9.2]])

def _pchip(x, xp, yp):
    """ Monotone piecewise cubic Hermite interpolation (Fritsch-Carlson)
        of scalar x on the points (xp, yp). Does not overshoot the data.
    """
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    h = np.diff(xp)
    delta = np.diff(yp) / h
    # Derivatives at the data points
    m = np.zeros_like(yp)
    for i in range(1, len(xp) - 1):
        if delta[i-1] * delta[i] > 0:
            w1 = 2*h[i] + h[i-1]
            w2 = h[i] + 2*h[i-1]
            m[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])
    for i, j, k in ((0, 0, 1), (-1, -1, -2)):  # one-sided end derivatives
        hj, hk = h[j], h[k]
        d = ((2*hj + hk)*delta[j] - hj*delta[k]) / (hj + hk)
        if np.sign(d) != np.sign(delta[j]):
            d = 0.0
        elif np.sign(delta[j]) != np.sign(delta[k]) and abs(d) > abs(3*delta[j]):
            d = 3*delta[j]
        m[i] = d
    i = min(max(np.searchsorted(xp, x) - 1, 0), len(xp) - 2)
    t = (x - xp[i]) / h[i]
    return ((2*t**3 - 3*t**2 + 1)*yp[i] + (t**3 - 2*t**2 + t)*h[i]*m[i]
            + (-2*t**3 + 3*t**2)*yp[i+1] + (t**3 - t**2)*h[i]*m[i+1])

def resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w, b=None):
    """ Return grid resistance with earth rods as per ENA EREC S34

        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L_E: length of horizontal electrode (m)
        L_R: length of vertical rod electrode (m)
        N: total number of rods
        S: seperation between rods (m)
        d: diameter of rod electrode (m)
        w: width of tape (m); used as b = w/pi when b is not given
        b: equivalent diameter of the grid conductor (m) in R12. Overrides w.
           S34 Appendix A and the worked examples (6.1.1, 6.5.1) use the
           conductor diameter (b = d for a round conductor; for a tape,
           b = perimeter/pi as in Formula P1). Appendix B.1.7 prints
           b = w/pi, which is the default.

        Factor k is interpolated from ENA EREC S34 Figure B.1, which covers
        0-72 rods; for N > 72, k is held at the chart end value (9.2).
    """
    N_max = _S34_FIG_B1_K[-1, 0]
    if N > N_max:
        warnings.warn('N = {} is beyond ENA EREC S34 Figure B.1 (0-{:g} rods);'
                      ' factor k held at {:g}'.format(N, N_max,
                                                     _S34_FIG_B1_K[-1, 1]))
    k = _pchip(min(N, N_max), _S34_FIG_B1_K[:,0], _S34_FIG_B1_K[:,1])

    r = sqrt(A/pi)
    R1 = rho/(4*r) + rho/L_E
    
    R_R = rho/(2*pi*L_R)*(log(8*L_R/d)-1)
    alpha = rho/(2*pi*R_R*S)
    R2 = R_R*(1+k*alpha)/N
    
    if b is None:
        b = w/pi
    R12 = R1 - rho/(pi*L_E)*(log(L_R/b)-1)
    
    R_E = (R1*R2 - R12**2)/(R1 + R2 - 2*R12)
    return R_E

## Touch & step potential fucntions

def _c_s(rho, rho_s, h_s):
    """ Surface layer derating factor, IEEE 80 Eq. (27) """
    return 1 - 0.09*(1-rho/rho_s)/(2*h_s+0.09)

def e_step_50(rho, rho_s, h_s, t_s):
    """ Tolerable step voltage, 50 kg body, as per IEEE 80 Eq. (29)

        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duration (s)
    """
    return (1000 + 6*_c_s(rho, rho_s, h_s)*rho_s)*0.116/sqrt(t_s)

def e_touch_50(rho, rho_s, h_s, t_s):
    """ Tolerable touch voltage, 50 kg body, as per IEEE 80 Eq. (32)

        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duration (s)
    """
    return (1000 + 1.5*_c_s(rho, rho_s, h_s)*rho_s)*0.116/sqrt(t_s)

def e_step_70(rho, rho_s, h_s, t_s):
    """ Step potential calculation as per IEEE 80
    
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    E_step_70 = (1000 + 6*_c_s(rho, rho_s, h_s)*rho_s)*0.157/sqrt(t_s)
    return E_step_70
    
def e_touch_70(rho, rho_s, h_s, t_s):
    """ Touch potential calculation as per IEEE 80
    
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    E_touch_70 = (1000 + 1.5*_c_s(rho, rho_s, h_s)*rho_s)*0.157/sqrt(t_s)
    return E_touch_70
    
def e_mesh_step(rho, Lc, Lp, A, Lx, Ly, D, Lr, Nr, d, h, Ig, Dm=None,
                rods_on_perimeter=True):
    """ Mesh and step voltage as per IEEE 80 Eqs. (80)-(94), for square,
        rectangular, L-shaped and other irregular grids

        Parameters:
            rho: resistivity (ohm-m)
            Lc: total length of horizontal grid conductor (m)
            Lp: peripheral length of the grid (m)
            A: area of the grid (m^2)
            Lx: maximum length of the grid in the x direction (m)
            Ly: maximum length of the grid in the y direction (m)
            D: spacing between parallel conductors (m)
            Lr: length of each earth rod (m)
            Nr: number of earth rods
            d: diameter of the grid conductors (m)
            h: grid burial depth (m)
            Ig: maximum grid current (A)
            Dm: maximum distance between any two points on the grid (m);
                defaults to sqrt(Lx^2 + Ly^2), i.e. n_d = 1 (square,
                rectangular and L-shaped grids)
            rods_on_perimeter: True for rods in the corners and along the
                perimeter (K_ii = 1, L_M from Eq. (91)); False for no rods
                or a few rods inside the grid only (K_ii from Eq. (82),
                L_M from Eq. (90))
        Return:
            Em: mesh voltage (V)
            Es: step voltage (V)
    """
    L_R = Lr*Nr  # Total length of all earth rods (m)
    h0 = 1  # Grid reference depth (m)
    if Dm is None:
        Dm = sqrt(Lx**2 + Ly**2)

    # Grid shape components, Eqs. (84)-(88). Each reduces to 1 for the
    # shapes where IEEE 80 sets it to 1.
    n_a = 2*Lc/Lp
    n_b = sqrt(Lp/(4*sqrt(A)))
    n_c = (Lx*Ly/A)**(0.7*A/(Lx*Ly))
    n_d = Dm/sqrt(Lx**2 + Ly**2)
    n = n_a*n_b*n_c*n_d

    Kh = sqrt(1+h/h0)
    perimeter_rods = Nr and rods_on_perimeter
    if perimeter_rods:
        Kii = 1
    else:
        Kii = 1/(2*n)**(2/n)

    Ki = 0.644 + 0.148*n
    Km = 1/(2*pi)*( log(D**2/(16*h*d) + (D+2*h)**2/(8*D*d) - h/(4*d))  \
                    + Kii/Kh*log(8/(pi*(2*n-1))) )
    Ks = 1/(pi)*( 1/(2*h) + 1/(D+h) + 1/D*(1-0.5**(n-2)) )

    if perimeter_rods:
        L_M = Lc + (1.55 + 1.22*(Lr/sqrt(Lx**2+Ly**2)))*L_R  # Eq. (91)
    else:
        L_M = Lc + L_R  # Eq. (90)
    Em = rho*Km*Ki*Ig/L_M
    Es = rho*Ks*Ki*Ig/(0.75*Lc + 0.85*L_R)

    return Em, Es

def e_mesh_step_grid(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig,
                     rods_on_perimeter=True):
    """ Mesh and step voltage of a rectangular grid as per IEEE 80

        Parameters:
            rho: resistivity (ohm-m)
            Lx: maximum length of the grid in the x direction (m)
            Ly: maximum length of the grid in the y direction (m)
            Lr: average earth rod length (m)
            Nx: number of grid conductors in x direction
            Ny: number of grid conductors in y direction
            Nr: number of earth rods
            d: diameter of the earth conductors (m)
            h: grid burial depth (m)
            Ig: maximum grid current (A)
            rods_on_perimeter: see e_mesh_step
        Return:
            Em: mesh voltage (V)
            Es: step voltage (V)
    """

    # Calculate geometric parameters
    A = Lx*Ly  # Area of grid
    Lc = Lx*Nx + Ly*Ny  # Total length of horizontal conductor in the grid (m)
    Lp = (Lx + Ly)*2  # Length of perimeter conductor (m)
    D = ( (Lx/(Ny-1)) + (Ly/(Nx-1)) )/2  # Spacing between parallel conductors
                                         # in the mesh (m)
    return e_mesh_step(rho, Lc, Lp, A, Lx, Ly, D, Lr, Nr, d, h, Ig,
                       rods_on_perimeter=rods_on_perimeter)

## Surface potential functions (ENA EREC S34 Appendix B.3)

def _s34_ke(h, d, D, n_A, n_B):
    """ Factor k_e of ENA EREC S34 Formula P1 """
    n = sqrt(n_A*n_B)
    return 1/pi*(0.5*log(h/d) + 1/(2*h) + 1/(0.5 + D) + 1/D*(1 - 0.5**(n-2)))

def _s34_kd(L_T, L_p):
    """ Factor k_d of ENA EREC S34 Formula P1 """
    return 0.7 + 0.3*L_T/L_p

def e_touch_external(rho, I_E, L_T, L_p, h, d, D, n_A, n_B):
    """ External touch potential 1 m beyond the edge of the grid electrode,
        ENA EREC S34 Formula P1

        rho: resistivity (ohm-m)
        I_E: current passing to ground through the electrode (A)
        L_T: total buried conductor length including connected rods (m)
        L_p: perimeter conductor length including connected rods (m)
        h: grid depth (m)
        d: equivalent conductor diameter = perimeter of conductor/pi (m)
        D: average spacing between parallel grid conductors (m)
        n_A, n_B: number of parallel grid conductors in each direction
    """
    return _s34_ke(h, d, D, n_A, n_B)*_s34_kd(L_T, L_p)*rho*I_E/L_T

def e_touch_fence(rho, I_E, L_T, L_p, h, d, D, n_A, n_B):
    """ External touch potential at a separately earthed fence 2 m from the
        grid electrode, ENA EREC S34 Formula P2 (k_f = 0.26 k_e)

        Parameters as e_touch_external. The divisor is L_T, as in worked
        example 6.1.5 (169 V) and as implied by "k_f is applied in place of
        k_e"; Formula P2 as printed divides by L_p.
    """
    return 0.26*e_touch_external(rho, I_E, L_T, L_p, h, d, D, n_A, n_B)

def e_touch_fence_peripheral(rho, I_E, L_T, L_p, h, d, S):
    """ External touch potential at a fence with a buried peripheral
        conductor 1 m outside it, ENA EREC S34 Formula P3, as printed:
        k_fe = 1/2 ln(h/d) - 1/4 ln((S^2 + 0.5^2)^2) + 1/4 ln(S^4 + S^2)

        rho, I_E, L_T, L_p, h, d: as e_touch_external
        S: distance between the outermost grid conductor and the next
           nearest parallel conductor (m)

        S34 has no worked example for P3, so it is not checked against a
        published value. The divisor is L_T, as for e_touch_fence.
    """
    k_fe = 0.5*log(h/d) - 0.25*log((S**2 + 0.5**2)**2) + 0.25*log(S**4 + S**2)
    return k_fe*_s34_kd(L_T, L_p)*rho*I_E/L_T

def e_step_external(rho, I_E, R_E, x):
    """ Step potential (1 m step) on the outside edge of a grid,
        ENA EREC S34 Formula P5

        rho: resistivity (ohm-m)
        I_E: electrode current (A)
        R_E: electrode resistance (ohm)
        x: distance from the centre of the grid (m), x > r = rho/(4 R_E)
    """
    r = rho/(4*R_E)
    return rho*I_E/(2*pi*r)*(asin(r/x) - asin(r/(x + 1)))

def surface_potential_hemisphere(rho, I, x):
    """ Surface potential at distance x (m) from a hemisphere electrode,
        ENA EREC S34 Formula P6.1
    """
    return rho*I/(2*pi*x)

def surface_potential_rod(rho, I, L_R, x):
    """ Surface potential at distance x (m) from a vertical rod of length
        L_R (m), ENA EREC S34 Formula P6.2
    """
    return rho*I/(2*pi*L_R)*log(L_R/x + sqrt(1 + L_R**2/x**2))

def surface_potential_grid(rho, I, R_E, x):
    """ Surface potential at distance x (m) from the centre of a buried grid
        of resistance R_E (ohm), ENA EREC S34 Formula P6.3; x > rho/(4 R_E)
    """
    r = rho/(4*R_E)
    return rho*I/(2*pi*r)*asin(r/x)

def contour_distance(A, V_x, U_E):
    """ Distance (m) from the edge of a grid to the surface potential contour
        V_x (V), ENA EREC S34 Formula P7

        A: grid area (m^2)
        V_x: contour potential (V), V_x < U_E
        U_E: earth potential rise (V)
    """
    if V_x >= U_E:
        raise ValueError('V_x must be lower than U_E')
    return sqrt(A/pi)*(1/sin(V_x*pi/(2*U_E)) - 1)

## Permissible touch and step potentials (ENA TS 41-24 Tables 1 and 2)

_TS4124_TIMES = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 5.0, 10.0)
_TS4124_TOUCH = {
    'bare_feet': (521, 462, 407, 313, 231, 166, 128, 106, 92, 84, 80, 76, 73,
                  71, 69, 67, 63, 60, 58, 57),
    'shoes_soil': (2070, 1808, 1570, 1179, 837, 578, 420, 332, 281, 250, 233,
                   219, 209, 200, 193, 188, 173, 162, 156, 153),
    'shoes_chippings_75': (2341, 2043, 1773, 1331, 944, 650, 471, 371, 314,
                           279, 259, 244, 232, 223, 215, 209, 192, 180, 173,
                           170),
    'shoes_chippings_150': (2728, 2379, 2064, 1548, 1095, 753, 544, 428, 361,
                            321, 298, 280, 266, 255, 246, 239, 220, 205, 198,
                            194),
    'hand_to_hand': (1114, 968, 836, 639, 484, 368, 276, 221, 191, 172, 161,
                     152, 146, 141, 137, 134, 125, 119, 115, 114),
}
_TS4124_STEP = {
    'bare_feet': (22753, 19763, 17077, 12715, 8905, 6044, 4290, 3320, 2770,
                  2434, 2249, 2098, 1992, 1897, 1823, 1771, 1616, 1503, 1442,
                  1412),
    'shoes_soil': (inf,)*8 + (21608, 19067, 17571, 16460, 15575, 14839,
                              14267, 13826, 12629, 11727, 11250, 11012),
    'shoes_chippings_75': (inf,)*8 + (24906, 21976, 20253, 18971, 17951,
                                      17103, 16445, 15936, 14557, 13517,
                                      12967, 12692),
    'shoes_chippings_150': (inf,)*10 + (24083, 22559, 21347, 20338, 19555,
                                        18951, 17311, 16074, 15420, 15092),
}

def _ts4124_lookup(table, t, surface, interpolate):
    if surface not in table:
        raise ValueError('surface must be one of ' + ', '.join(table))
    values = table[surface]
    times = _TS4124_TIMES
    if t <= 0:
        raise ValueError('t must be positive')
    if t < times[0]:
        warnings.warn('t = {} s is below the shortest tabulated time ({} s);'
                      ' the {} s value is returned'.format(t, times[0],
                                                           times[0]))
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    i = int(np.searchsorted(times, t, side='right'))  # times[i-1] <= t < times[i]
    if t == times[i-1]:
        return float(values[i-1])
    if interpolate == 'conservative':
        return float(values[i])  # next longer time, lower limit
    elif interpolate == 'linear':
        v0, v1 = values[i-1], values[i]
        if v0 == inf:
            return inf
        return float(v0 + (v1 - v0)*(t - times[i-1])/(times[i] - times[i-1]))
    raise ValueError("interpolate must be 'conservative' or 'linear'")

def touch_limit_ts4124(t, surface='shoes_chippings_75',
                       interpolate='conservative'):
    """ Permissible touch potential (V), ENA TS 41-24 Table 1

        t: fault clearance time (s)
        surface: 'bare_feet', 'shoes_soil' (shoes on soil or outdoor
            concrete), 'shoes_chippings_75', 'shoes_chippings_150' (or dry
            concrete), 'hand_to_hand'
        interpolate: between tabulated times, 'conservative' returns the
            value at the next longer tabulated time; 'linear' interpolates
            linearly. Times of 10 s or more return the >= 10 s value.
    """
    return _ts4124_lookup(_TS4124_TOUCH, t, surface, interpolate)

def step_limit_ts4124(t, surface='shoes_chippings_75',
                      interpolate='conservative'):
    """ Permissible step potential (V), ENA TS 41-24 Table 2

        t, interpolate: as touch_limit_ts4124
        surface: 'bare_feet', 'shoes_soil', 'shoes_chippings_75',
            'shoes_chippings_150'

        Returns math.inf where the table states the limit cannot
        foreseeably be exceeded (25 kV or greater).
    """
    return _ts4124_lookup(_TS4124_STEP, t, surface, interpolate)

## Fault current functions

def fault_current(Un, c, Z):
    """ Fault current calculation as per IEC-60909
    
        Un: Nominal line-line voltage (V)
        c: Voltage factor
        Z: Complex impedence
    """
    return c*Un/(sqrt(3)*abs(Z))

def max_current_density(rho, t, method='S34'):
    """ Electrode maximum current density

        rho: resistivity (ohm-m)
        t: Loading time (s)
        method: 'S34'     - ENA EREC S34 Formula C2, 1e3 sqrt(57.7/(rho t)),
                            as used for ENA TS 41-24 Table 8 (default)
                'IEEE142' - IEEE Std 142-2007 Eq. (4.2), I = 1140 d/sqrt(rho t)
                            A per metre of rod (d in mm, rho in ohm-cm),
                            divided by the rod surface area per metre
                            (pi d/1000 m^2): 1.14e5/pi / sqrt(rho t)
                            = 3.63e4/sqrt(rho t), rho in ohm-m.
                            Eq. (4.2) is stated for rods, to prevent
                            smoking electrodes; it is about 4.8 times the
                            'S34' and 'IS3043' limits.
                'IS3043'  - IS 3043:2018 clause 15.3 b),
                            7.57e3/sqrt(rho t) (0.34 % below 'S34')
        return (A/m2)
    """
    if method == 'IEEE142':
        return 1.14e5/pi/sqrt(rho*t)
    elif method == 'IS3043':
        return 7.57e3/sqrt(rho*t)
    elif method == 'S34':
        return 1e3*sqrt(57.7/(rho*t))
    raise ValueError("method must be 'IEEE142', 'S34' or 'IS3043'")

def current_ratio(rho, R, C, Un, A, L):
    """ Current ratio by C factor method as per ENA EREC S34
    
        rho: resistivity (ohm-m)
        R: combined sorce and destination resistance
        C: C factor
        Un: System voltage (kV)
        A: Cross sectional area (mm2)
        L: Cable length (km)
    """
    return C/(A+9*Un) / sqrt((C/(A+9*Un) + R/L)**2 + 0.6*(rho/(A*Un))**0.1)

# Convinience functions

def resistance_parallel(*res):
    """ Return effective of parallel resistances """
    R = 0
    for r in res:
        R += r**-1
    return R**-1
    
def rectangle(loc, x_l, y_l):
    """ Return a rectangular polygon
        loc: (x,y) coordinates of start location
        x_l: length of rectangle along x
        y_l: width of rectangle along y
    """
    
    x,y = loc
    return ((x,y), (x+x_l, y), (x+x_l, y+y_l), (x, y+y_l), (x,y))

## CALCULATION ELEMENT CLASSES


class DiscreteElement:
    """Base class for all unit elements"""
    
    def __init__(self, loc, rho):
        self.loc = np.array(loc)
        self.rho = rho
        
    def pot_coeff_self(self, loc):
        """ Function to compute self potencial coefficient"""
        pass
    
    def pot_coeff(self, loc, mirror=False):
        """ Function to compute mutual potencial coefficient"""
        pass

    
class DiscreteElementCylindrical(DiscreteElement):
    """ 1D approximation for cylindrical elememt
        Uses radial current distribution upto critical distance and 
        spherical distribution beyond critical distance
    """
    
    def __init__(self, loc, rho, radius, length):
        DiscreteElement.__init__(self, loc, rho)
        self.radius = radius
        self.length = length
        self.crit_d = self.length  # Distance upto which radial current 
                                   # distribution holds

    def pot_coeff_self(self):
        coeff = self.rho/(4*pi)*(1/self.crit_d \
                                 + 2*log(self.crit_d/self.radius)/self.length)
        return coeff
        
    def pot_coeff(self, loc, mirror=False):
        crit_d = 2*self.length  # Distance upto which radial current 
                                # distribution holds
        if mirror == False:
            dist = norm(loc - self.loc)
        else:
            mirror_loc = np.copy(self.loc)
            mirror_loc[2] = -mirror_loc[2]
            dist = norm(loc - mirror_loc)
        
        if dist <= self.radius:
            raise ValueError('Coordinates within element boundary')
        elif dist < self.crit_d:  # assume radial current distribution
            coeff = self.rho/(4*pi)*(1/self.crit_d \
                                     + 2*log(self.crit_d/dist)/self.length)
        else:  # assume spherical current distribution
            coeff = self.rho/(4*pi*dist)
        return coeff


class DiscreteElementPlate(DiscreteElement):
    """ Circular plate elememt
        Uses equations from the paper:
        Ground impedance of cylindrical metal plate buried in homogeneous earth
        - Slavko Vujević, Zdenko Balaž and Dino Lovrić
    """
    
    def __init__(self, loc, rho, area, normal):
        DiscreteElement.__init__(self, loc, rho)
        self.area = area
        self.normal = np.array(normal)
        self.radius = sqrt(area/pi)
        
    def pot_coeff_self(self):
        a = self.radius
        coeff = self.rho/(8*a)
        return coeff
        
    def pot_coeff(self, loc, mirror=False):
        a = self.radius
        if mirror == False:
            d = loc - self.loc
        else:
            mirror_loc = np.copy(self.loc)
            mirror_loc[2] = -mirror_loc[2]
            d = loc - mirror_loc
        z = norm(self.normal * d)
        r = norm(d - (self.normal * d))
        A = r**2 + z**2 - a**2
        alpha = sqrt((A + sqrt(A**2 + 4 * a**2 * z**2))/2)
        coeff = self.rho/(4*pi*a)*atan(a/alpha)
        return coeff


# Deprecated misspelled class names, kept for backward compatibility
DescreteElement = DiscreteElement
DescreteElementCylindrical = DiscreteElementCylindrical
DescreteElementPlate = DiscreteElementPlate


## NETWORK ELEMENT CLASSES


class NetworkElement:
    """Base class for all earth electrod primitives"""
    
    def __init__(self, loc, rho):
        self.loc = np.array(loc)
        self.rho = rho
        self.elements = []
        
    def discretise(self):
        self.elements = []
        
    def get_discrete_elements(self):
        return self.elements

    # Deprecated misspelled names, kept for backward compatibility
    def descretise(self, *args, **kwargs):
        warnings.warn('descretise() is deprecated; use discretise()',
                      DeprecationWarning, stacklevel=2)
        return self.discretise(*args, **kwargs)

    def get_descrete_elements(self):
        warnings.warn('get_descrete_elements() is deprecated; use '
                      'get_discrete_elements()', DeprecationWarning,
                      stacklevel=2)
        return self.get_discrete_elements()
        
        
class NetworkElementPlate(NetworkElement):
    
    def __init__(self, loc, rho, w, h, n_cap=[1,0,0], h_cap=[0,0,1]):
        NetworkElement.__init__(self, loc, rho)
        self.w = w
        self.h = h
        self.n_cap = np.array(n_cap) / norm(n_cap)
        self.h_cap = np.array(h_cap) / norm(h_cap)
        self.w_cap = np.cross(self.h_cap, self.n_cap)
        
    def discretise(self, size):
        NetworkElement.discretise(self)
        nw = int(self.w/size/2)
        nh = int(self.h/size/2)
        delta_w = self.w/(2*nw) * self.w_cap
        delta_h = self.h/(2*nh) * self.h_cap
        area = self.w/nw * self.h/nh / 4
        for i in range(-nw, nw):
            for j in range(-nh, nh):
                loc = (self.loc + delta_w/2 + delta_h/2) + delta_w*i + delta_h*j
                element = DiscreteElementPlate(loc, self.rho, area, self.n_cap)
                self.elements.append(element)


class NetworkElementStrip(NetworkElement):
    
    def __init__(self, loc, rho, w, loc_end):
        NetworkElement.__init__(self, loc, rho)
        self.w = w
        self.loc_end = np.array(loc_end)
        
    def discretise(self, size):
        NetworkElement.discretise(self)
        length = norm(self.loc - self.loc_end)
        n = int(length/size)
        delta = length/n
        delta_vec = (self.loc_end - self.loc)/length*delta
        n_cap = [0,0,1]
        area = self.w * norm(delta_vec)
        for i in range(0, n):
            loc = (self.loc + delta_vec/2) + delta_vec * i
            element = DiscreteElementPlate(loc, self.rho, area, n_cap)
            self.elements.append(element)
            
    
class NetworkElementPipe(NetworkElement):
    
    def __init__(self, loc, rho, radius, loc_end):
        NetworkElement.__init__(self, loc, rho)
        self.radius = radius
        self.loc_end = np.array(loc_end)
        
    def discretise(self, size):
        NetworkElement.discretise(self)
        
        def perp_vec(vec): 
            a,b,c = vec
            if a != 0 or b != 0:
                v1 = np.array((b,-a,0))
            else:
                v1 = np.array((0,-c,b))
            return v1 / norm(v1)
            
        length = norm(self.loc - self.loc_end)
        n = int(length/size)
        delta = length/n
        delta_vec = (self.loc_end - self.loc)/length*delta
        h_cap = delta_vec / norm(delta_vec)
        n_cap = perp_vec(h_cap)
        area = 2*pi*self.radius*delta/2  # Plate has two sides so eq area halfed
        for i in range(0, n):
            loc = (self.loc + delta_vec/2) + delta_vec * i
            element = DiscreteElementPlate(loc, self.rho, area, n_cap)
            self.elements.append(element)
            
            
## NETWORK CLASS
        
    
class Network:
    """Class for network definition and solving"""
    
    def __init__(self, rho, Ig):
        """ rho: resistivity
            Ig: array of grid current injection 
        """
        self.rho = rho
        self.Ig = [Ig]
        
        self.elements = [[]]
        self.clear_model()
        
    @property
    def descrete_elements(self):
        """ Deprecated alias of discrete_elements """
        warnings.warn('descrete_elements is deprecated; use discrete_elements',
                      DeprecationWarning, stacklevel=2)
        return self.discrete_elements

    def clear_model(self):
        """ Clear calculation variables """
        self.discrete_elements = []
        self.subnet_sizes = []
        self.A = None
        self.B = None
        self.X = None
        self.I = None
        self.V = None
        self.XX = None
        self.YY = None
        self.Vg = None
        
    # Model formulation functions
        
    def __add__(self, obj):
        retobj = Network(self.rho, 0)
        retobj.Ig = self.Ig + obj.Ig
        retobj.elements = self.elements + obj.elements
        return retobj
    
    def add_subnet(self, Ig):
        self.Ig.append(Ig)
        self.elements.append([])
        self.clear_model()

    def add_strip(self, loc_start, loc_end, w):
        """ Add strip element
        
            loc_start: Start coordinates of strip
            loc_end: End coordinates of strip
            w: Strip width
        """
        element = NetworkElementStrip(np.array(loc_start), 
                                      self.rho, w, np.array(loc_end))
        self.elements[-1].append(element)
        
    def add_rod(self, loc, radius, length):
        """ Add a vertical rod/ pipe element
        
            loc: Coordinates of pipe (z coordinate defines the rod top)
            radius: Rod radius
            length: Rod length
        """
        loc_start = np.array(loc)
        loc_end = np.array(loc) + np.array([0,0,-length])
        element = NetworkElementPipe(loc_start, self.rho, radius, loc_end)
        self.elements[-1].append(element)
    
    def add_mesh(self, loc, Lx, Ly, Nx, Ny, w):
        """ Add a uniform mesh earthing grid
        
            loc: Start coordinates of grid
            Lx: maximum length of the grid in the x direction (m)
            Ly: maximum length of the grid in the y direction (m)
            Nx: number of grid conductors in x direction
            Ny: number of grid conductors in y direction
            w: Strip width
        """
        
        for i in range(Nx):
            loc_start = np.array(loc) + np.array([0, i*Ly/(Nx-1), 0])
            loc_end = loc_start + np.array([Lx, 0, 0])
            self.add_strip(loc_start, loc_end, w)
        # Add second row offset by 2*w to avoid computation errors
        for i in range(Ny):
            loc_start = np.array(loc) + np.array([i*Lx/(Ny-1), 0, 0])
            loc_end = loc_start + np.array([0, Ly, 0])
            self.add_strip(loc_start, loc_end, w)
            
    def add_plate(self, loc, w, h, n_cap=[1,0,0], h_cap=[0,0,1]):
        """ Add plate element
        
            loc: Coordinates of pipe (z coordinate defines the rod top)
            w: Width of plate
            h: Height of plate
            n_cap: Normal vector perpendicular to plate
            h_cap: Vector in the direction of plate height
        """
        element = NetworkElementPlate(loc, self.rho, w, h, n_cap, h_cap)
        self.elements[-1].append(element)
        
    # Solution functions
    
    def mesh_geometry(self, desc_size=0.2):
        """Descretise geometry"""
        
        if not self.elements[-1]:
            raise Exception('No problem geometry found')
            
        self.discrete_elements = []
        self.subnet_sizes = []
        
        # Descretise Geometry
        for subnet in self.elements:
            subnet_discrete_elements = []
            for element in subnet:
                element.discretise(desc_size)
                discrete_elements = element.get_discrete_elements()
                subnet_discrete_elements += discrete_elements
            self.discrete_elements += subnet_discrete_elements
            self.subnet_sizes.append(len(subnet_discrete_elements))
            
    def generate_model(self, desc_size=0.25):
        """Formulate system of equations"""
        
        if not self.elements[-1]:
            raise Exception('No problem geometry found')
        
        self.mesh_geometry(desc_size)
        
        n = len(self.discrete_elements)
        k = len(self.elements)
        
        #
        # Total k subnets
        #
        # AX = B ; the system of equation to be solved (n1+n2+...+nk + k) x (n1+n2+...+nk + k)
        #
        # sum{(Pij+Pij')*Ii} - V1 = 0  -- n1 equations
        # sum{(Pij+Pij')*Ii} - V2 = 0  -- n2 equations
        # ...
        # sum{(Pij+Pij')*Ii} - Vk = 0  -- nk equations
        # 
        # I11 + I12 + ... + I1n1 + 0*V1 + 0*V2 ... + 0*Vk = Ig1
        # I21 + I22 + ... + I2n2 + 0*V1 + 0*V2 ... + 0*Vk = Ig2
        # ...
        # Ik1 + Ik2 + ...  + Iknk + 0*V1 + 0*V2 ... + 0*Vk = Igk
        #
        # X = [I11 I12 ... Iij ... In V1 V2 ... Vk]
        #
        # A is of form
        # | nxn | nxk |        | A1 | A2 |
        # | kxn | kxk |        | A3 | 0 |
        #
        # B is of form
        # | nx1 |              | 0  |
        # | kx1 |              | Ig |
        #
        
        # B matrix
        self.B = np.zeros((n+k, 1))
        self.B[n:(n+k), 0] = self.Ig
        
        # A matrix
        self.A = np.zeros((n+k, n+k))
        
        # A1 - P(n x n)
        for i, element_cur in enumerate(self.discrete_elements):
            for j, element_rem in enumerate(self.discrete_elements):
                if i == j:  # set diag terms
                    self.A[i,i] = element_cur.pot_coeff_self() \
                                  +  element_cur.pot_coeff(element_cur.loc, 
                                                           mirror=True)
                else:  # set off diag terms
                    self.A[i,j] = element_cur.pot_coeff(element_rem.loc) \
                                  + element_cur.pot_coeff(element_rem.loc, 
                                                          mirror=True)
        
        # A2 - V(n x k)
        A2 = np.zeros((n, k))
        cum_index = 0
        for slno, n_subnet in enumerate(self.subnet_sizes):
            A2[cum_index:(cum_index+n_subnet), slno] = -1
            cum_index += n_subnet
        self.A[0:n, n:(n+k)] = A2
        
        # A3 - 1(k x n)
        A3 = -A2.T
        self.A[n:(n+k), 0:n] = A3
        
    def generate_model_fast(self, desc_size=0.25):
        """ Formulate system of equations
            Assume circular plate discretisation for fast calculation 
        """
        
        if not self.elements[-1]:
            raise Exception('No problem geometry found')
        
        self.mesh_geometry(desc_size)
        
        n = len(self.discrete_elements)
        k = len(self.elements)
        
        #
        # Total k subnets
        #
        # AX = B ; the system of equation to be solved (n1+n2+...+nk + k) x (n1+n2+...+nk + k)
        #
        # sum{(Pij+Pij')*Ii} - V1 = 0  -- n1 equations
        # sum{(Pij+Pij')*Ii} - V2 = 0  -- n2 equations
        # ...
        # sum{(Pij+Pij')*Ii} - Vk = 0  -- nk equations
        # 
        # I11 + I12 + ... + I1n1 + 0*V1 + 0*V2 ... + 0*Vk = Ig1
        # I21 + I22 + ... + I2n2 + 0*V1 + 0*V2 ... + 0*Vk = Ig2
        # ...
        # Ik1 + Ik2 + ...  + Iknk + 0*V1 + 0*V2 ... + 0*Vk = Igk
        #
        # X = [I11 I12 ... Iij ... In V1 V2 ... Vk]
        #
        # A is of form
        # | nxn | nxk |        | A1 | A2 |
        # | kxn | kxk |        | A3 | 0 |
        #
        # B is of form
        # | nx1 |              | 0  |
        # | kx1 |              | Ig |
        #
        
        # B matrix
        self.B = np.zeros((n+k, 1))
        self.B[n:(n+k), 0] = self.Ig
        
        # A matrix
        self.A = np.zeros((n+k, n+k))
        
        # A1 - P(n x n)
        
        # Form location array
        LOC = np.zeros((n,3))
        LOCm = np.zeros((n,3))
        for slno, element in enumerate(self.discrete_elements):
            mirror_loc = np.copy(element.loc)
            mirror_loc[2] = -mirror_loc[2]
            LOC[slno] = element.loc
            LOCm[slno] = mirror_loc
            
        for slno, element in enumerate(self.discrete_elements):
        
            N = np.repeat(element.normal[np.newaxis, :], n, axis=0)
            a = element.radius
            
            # Form distance array
            loc = np.repeat(LOC[slno][np.newaxis, :], n, axis=0)
            loc_m = np.repeat(LOCm[slno][np.newaxis, :], n, axis=0)
            DD = LOC - loc  # Distance from current point to remote point
            DDm = LOC - loc_m  # Distance from mirror point to remote point
            
            # Get cylindrical coordinates
            Z = norm(N * DD, axis=1)
            R = norm(DD - (N * DD), axis=1)
            Zm = norm(N * DDm, axis=1)
            Rm = norm(DDm - (N * DDm), axis=1)
            
            # Get coefficiencts
            A_ = R**2 + Z**2 - a**2
            ALPHA = np.sqrt((A_ + np.sqrt(A_**2 + 4 * a**2 * Z**2))/2)
            COEF = self.rho/(4*pi*a)*np.arctan(a/ALPHA)
            A_m = Rm**2 + Zm**2 - a**2
            ALPHAm = np.sqrt((A_m + np.sqrt(A_m**2 + 4 * a**2 * Zm**2))/2)
            COEFm = self.rho/(4*pi*a)*np.arctan(a/ALPHAm)
            
            # Update A1
            self.A[slno, 0:n] = (COEF + COEFm)
        
        # A2 - V(n x k)
        A2 = np.zeros((n, k))
        cum_index = 0
        for slno, n_subnet in enumerate(self.subnet_sizes):
            A2[cum_index:(cum_index+n_subnet), slno] = -1
            cum_index += n_subnet
        self.A[0:n, n:(n+k)] = A2
        
        # A3 - 1(k x n)
        A3 = -A2.T
        self.A[n:(n+k), 0:n] = A3
        
    def solve_model(self):
        """Solve system of equations"""
        
        if not self.discrete_elements:
            raise Exception('No model generated')
        
        k = len(self.elements)
        
        self.X = solve(self.A, self.B)
        self.I = self.X[:-k,0]
        self.V = self.X[-k:,0]
    
    def get_resistance(self):
        """Get overall resistance of grid"""
        
        if self.X is None:
            raise Exception('Model not solved')
        
        res = self.V / self.Ig
        return np.round(res, 3)
    
    def gpr(self):
        """ Calculate ground potential rise of grid
        """
        
        if self.V is None:
            raise Exception('Model not solved')
            
        return np.round(self.V, 3)
    
    def get_point_potential(self, loc):
        """ Get potential at a fixed location
        
            loc: Location for calculation
        """
        if self.X is None:
            raise Exception('Model not solved')
            
        v = 0
        for slno, element in enumerate(self.discrete_elements):
            coeff = element.pot_coeff(loc) + element.pot_coeff(loc, mirror=True)
            v += coeff * self.I[slno]
        return v
    
    def solve_surface_potential(self, grid=(20,20), 
                                xlim=(-20, 20), ylim=(-20, 20)):
        """ Solve for surface potentials on a uniform grid
        
            grid: (nx, ny) -> number of calculation points along x & y 
            xlim: x limits for calculation
            ylim: y limits for calculation
        """
        
        if self.X is None:
            raise Exception('Model not solved')
            
        X, Y = np.meshgrid(np.linspace(*xlim, grid[0]), 
                           np.linspace(*ylim, grid[1]))
        V = X*0.0
        for i, (XX, YY) in enumerate(zip(X, Y)):
            for j, (x, y) in enumerate(zip(XX, YY)):
                loc = np.array([x,y,0])
                V[i, j] = self.get_point_potential(loc)
        self.XX = X
        self.YY = Y
        self.Vg = V
        
    def solve_surface_potential_fast(self, grid=(20,20), 
                                     xlim=(-20, 20), ylim=(-20, 20), 
                                     save_results=True):
        """ Solve for surface potentials on a uniform grid
            Assumes sperical potential distribution of elements for fast 
            calculation 

            grid: (nx, ny) -> number of calculation points along x & y 
            xlim: x limits for calculation
            ylim: y limits for calculation
            save_results: True: Save results
                          False: Return voltage array on instead of saving the 
                                 results for external analysis
        """
        
        if self.X is None:
            raise Exception('Model not solved')
        
        # Form calculation mesh
        XX, YY = np.meshgrid(np.linspace(*xlim, grid[0]), 
                             np.linspace(*ylim, grid[1]))
        LOC = np.stack((XX, YY, np.zeros(XX.shape)), axis=2)
        VV = np.zeros(XX.shape)
        
        # Calculate distance array
        for slno, element in enumerate(self.discrete_elements):
            (x, y, z) = element.loc
            ones = np.ones(XX.shape[0:2])
            LOCe = np.stack((ones*x, ones*y, ones*z), axis=2)
            DDD = norm(LOC - LOCe, axis=2)
            Ie = self.I[slno]
            # Symmetry allows doubling of voltage on account of mirror geometry
            VV += Ie * self.rho/(4*pi*DDD) * 2 
        
        if save_results:
            self.XX = XX
            self.YY = YY
            self.Vg = VV
        else:
            return VV
        
    def mesh_voltage(self, polygon_points, mesh_no=10, 
                     plot=False, plot_type='fill', levels=10, grid_spacing=1,
                     subnet=0):
        """ Find mesh voltage in the passed polygon
        
            Parameters:
                polygon: Corner points of polygon (m)
                mesh_no: Number of points per m for calculation
                plot: Plot mesh voltages
                plot_type: May take values 'contour', 'fill', 'values'
                levels: Number of levels for contour/ fill plot
                grid_spacing: Spacing of plot grid
            Returns:
                (location of max mesh voltage, maximum mesh voltage)
        """
        polygon_array = np.array(polygon_points)
        polygon = path.Path(polygon_array)
        
        # Find limits of polygon and find voltages at corresponding mesh
        x0 = np.min(polygon_array[:,0])
        x1 = np.max(polygon_array[:,0])
        y0 = np.min(polygon_array[:,1]) + 1/mesh_no  # Hack to cutoff transition 
                                                     # from 0 to first point
        y1 = np.max(polygon_array[:,1])
        xlim = (x0,x1)
        ylim = (y0,y1)
        x_samples = int((x1-x0)*mesh_no) + 1
        y_samples = int((y1-y0)*mesh_no) + 1
        grid = (x_samples, y_samples)
        V = self.solve_surface_potential_fast(grid, xlim, ylim, 
                                              save_results=False)
        
        # Prepare mask with polygon
        xx, yy = np.meshgrid(np.linspace(x0, x1, x_samples),
                             np.linspace(y0, y1, y_samples))
        index_array = np.hstack((xx.flatten()[:,np.newaxis], 
                                 yy.flatten()[:,np.newaxis]))
        flags = polygon.contains_points(index_array).reshape((grid[1], grid[0]))
             
        # Find mesh voltage from masked voltage
        V_masked = (self.gpr()[subnet] - V) * flags  #TODO
        v_max = np.max(V_masked)
        index_max = np.where(V_masked == v_max)
        loc_max = (xx[index_max][0], yy[index_max][0])
        
        # Plot if called for
        if plot:
            self.plot_surface_potential(xlim, ylim, plot_type=plot_type, 
                                        plot_data=(xx, yy, V_masked), 
                                        title='Mesh voltage profile',
                                        levels=levels, 
                                        grid_spacing=grid_spacing)
                                    
        return np.round(loc_max, 3), np.round(v_max, 3)
        
    def step_voltage(self, polygon_points, mesh_no=14, step_size=1, 
                     plot=False, plot_type='fill', levels=10, grid_spacing=1):
        """ Find step voltage in the passed polygon
            Function shifts voltage mesh grids in 8 directions for evaluation
            of step voltage at various points
            
            Parameters:
                polygon: Corner points of polygon (m)
                mesh_no: Number of points per m for calculation
                step_size: Step size for step voltage calculation (m)
                plot: Plot mesh voltages
                plot_type: May take values 'contour', 'fill', 'values'
                levels: Number of levels for contour/ fill plot
                grid_spacing: Spacing of plot grid
            Returns:
                (location of max step voltage, maximum step voltage)
        """
        polygon_array = np.array(polygon_points)
        polygon = path.Path(polygon_array)
        offset = step_size  # Offset for computation 
        
        # Find limits of polygon and find voltages at corresponding mesh
        x0 = np.min(polygon_array[:,0]) - offset
        x1 = np.max(polygon_array[:,0]) + offset
        y0 = np.min(polygon_array[:,1]) - offset
        y1 = np.max(polygon_array[:,1]) + offset
        xlim = (x0, x1)
        ylim = (y0, y1)
        x_samples = int((x1-x0)*mesh_no) + 1 + offset*2
        y_samples = int((y1-y0)*mesh_no) + 1 + offset*2
        grid = (x_samples, y_samples)
        V = self.solve_surface_potential_fast(grid, xlim, ylim, 
                                              save_results=False)
        
        # Prepare mask with polygon
        xx, yy = np.meshgrid(np.linspace(x0, x1, x_samples),
                             np.linspace(y0, y1, y_samples))
        index_array = np.hstack((xx.flatten()[:,np.newaxis], 
                                 yy.flatten()[:,np.newaxis]))
        flags = polygon.contains_points(index_array).reshape((grid[1], grid[0]))
        
        # Compute step voltage array in various directions
        directions = np.array([(1,0) , (0,1),
                               (0.707,0.707), (-0.707, 0.707),
                               (0.5, 0.866), (-0.5, 0.866),
                               (0.866, 0.5), (-0.866, 0.5)])
        step_array = np.round(directions * mesh_no * step_size)
        V_steps = np.zeros((y_samples, x_samples, directions.shape[0]))
        for slno, shift_values in enumerate(step_array.astype('i')):
            x_shift, yshift = shift_values
            V0 = np.roll(V, (-x_shift, -yshift), axis=(0, 1))
            V1 = np.roll(V, (x_shift, yshift), axis=(0, 1))
            V_steps[:,:,slno] = np.maximum(np.absolute(V0-V), np.absolute(V1-V))
            
        V_step = np.max(V_steps, axis=2)
        
        # Mask required values
        V_masked = V_step * flags
        v_max = np.max(V_masked)
        index_max = np.where(V_masked == v_max)
        loc_max = (xx[index_max][0], yy[index_max][0])
        
        # Plot if called for
        if plot:
            self.plot_surface_potential(xlim, ylim, plot_type=plot_type, 
                                        plot_data=(xx, yy, V_masked), 
                                        title='Step voltage profile',
                                        levels=levels, 
                                        grid_spacing=grid_spacing)
        return np.round(loc_max, 3), np.round(v_max, 3)
        
    def plot_surface_potential(self, xlim=(-20, 20), ylim=(-20, 20), 
                               plot_type='fill', levels=10, grid_spacing=1, 
                               plot_data=None, 
                               title='Surface potential distribution'):
        """ Plot surface potential evaluated using solve_surface_potential(...) 
            or solve_surface_potential_fast(...)
        
            xlim: x limits for plot
            ylim: y limits for plot            
            plot_type: May take values 'contour', 'fill', 'values'
            levels: Number of levels for contour/ fill plot
            grid_spacing: Spacing of plot grid
            plot_data: (xx, yy, v) -> use custom plot data
            title: Display title
        """
        
        if plot_data is None:
            if self.Vg is None:
                raise Exception('Surface potential not solved')
            XX = self.XX
            YY = self.YY
            V = self.Vg
        else:
            XX, YY, V = plot_data
            
        ax = plt.figure().add_subplot()
        
        # Plot surface potential
        if plot_type == 'fill':
            contour_plot = ax.contourf(XX, YY, V, levels, cmap="plasma")
            cbar = plt.colorbar(contour_plot)
            # cbar.set_label('Volts', loc='bottom')
        elif plot_type == 'values':
            ax.scatter(XX, YY, color='blue', marker='.')  # plot points
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    value = round(V[i,j])
                    ax.annotate(str(value),xy=(XX[i,j], YY[i,j]))
        elif plot_type == 'contour':
            contour_plot = ax.contour(XX, YY, V, levels, cmap="plasma")
            ax.clabel(contour_plot, inline=True)

        
        # Plot problem geometry as lines
        styler = plot_cycler()
        for subnet, style in zip(self.elements, styler):
            for element in subnet:
                if isinstance(element, NetworkElementStrip) \
                   or isinstance(element, NetworkElementPipe):
                    start = element.loc
                    end = element.loc_end
                    X = [start[0], end[0]]
                    Y = [start[1], end[1]]
                    ax.plot(X, Y, **style, linewidth=eplot_linewidth)  # plot line
                if isinstance(element, NetworkElementPlate):
                    c1 = element.loc - element.w_cap*element.w/2 \
                         - element.h_cap*element.h/2
                    c2 = c1 + element.w_cap * element.w
                    c3 = c2 + element.h_cap * element.h
                    c4 = c3 - element.w_cap * element.w
                    X = [c1[0], c2[0], c3[0], c4[0], c1[0]]
                    Y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
                    ax.plot(X, Y, **style, linewidth=eplot_linewidth)  # plot line
        
        ax.set_xticks(np.arange(*xlim, grid_spacing))
        ax.set_yticks(np.arange(*ylim, grid_spacing))
        ax.grid()
        ax.set(xlim=xlim, ylim=ylim, xlabel='X', ylabel='Y')
        plt.title(title)
        plt.show()
    
    def plot_geometry_3d(self, xlim=(-20, 20), ylim=(-20, 20), zlim=(-5, 5), 
                         ground=True, ground_pot=False, 
                         current_distribution=False, 
                         normal=False, normal_scale=0.5, 
                         title='Problem geometry'):
        """ Display problem geometry and solutions in 3D
           
            xlim: x limits for plot
            ylim: y limits for plot   
            zlim: z limits for plot
            ground: Display ground
            ground_pot: Plot surface potential
            current_distribution: Plot current distribution
            normal: Show discrete element normal vectors
            normal_scale: Discrete element normal vector scale
            title: Display title
        """
        
        if ((ground_pot is True) or (current_distribution is True) \
           or (normal is True)) and self.X is None:
            raise Exception('Model not solved')
            
        ax = plt.figure().add_subplot(projection='3d')
        
        if ground:
            # Plot XY plane
            X, Y = np.meshgrid(np.linspace(*xlim), np.linspace(*ylim))
            Z = X*0
            ax.plot_surface(X, Y, Z, alpha=0.1, color='blue')
        
        if ground_pot:
            # Plot surface voltage contour
            contour_plot = ax.contourf(self.XX, self.YY, self.Vg, zdir='z', 
                                       offset=0, cmap="plasma", alpha=0.7)
            cbar = plt.colorbar(contour_plot, pad=0.1)
            # cbar.set_label('Volts', loc='bottom')
            
        # Plot direction vectors for discrete elements
        if normal:
            X = []
            Y = []
            Z = []
            U = []
            V = []
            W = []    
            for desc_element in self.discrete_elements:
                if isinstance(desc_element, DiscreteElementPlate):
                    x, y, z = desc_element.loc
                    u, v, w = desc_element.normal
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    U.append(u*normal_scale)
                    V.append(v*normal_scale)
                    W.append(w*normal_scale)
            ax.quiver(X, Y, Z, U, V, W)  # quiver
        
        # Set plot general properties    
        ax.set(xlim=xlim, ylim=ylim, zlim=zlim,
               xlabel='X', ylabel='Y', zlabel='Z')
        plt.title(title)
        plt.show()
        
        if current_distribution:
            # Plot problem geometry with current as weight
            X = []
            Y = []
            Z = []
            for slno, element in enumerate(self.discrete_elements):
                loc = element.loc
                X.append(loc[0])
                Y.append(loc[1])
                Z.append(loc[2])
            scat_plot = ax.scatter(X, Y, Z, c=self.I, s=5, cmap='viridis')
            cbar = plt.colorbar(scat_plot, pad=0.1)
            # cbar.set_label('Amps', loc='bottom')
        else:
            # Plot problem geometry as lines
            styler = plot_cycler()
            for subnet, style in zip(self.elements, styler):
                for element in subnet:
                    if isinstance(element, NetworkElementStrip) \
                       or isinstance(element, NetworkElementPipe):
                        start = element.loc
                        end = element.loc_end
                        X = [start[0], end[0]]
                        Y = [start[1], end[1]]
                        Z = [start[2], end[2]]
                        ax.plot(X, Y, Z, **style, linewidth=eplot_linewidth)  # plot line
                    if isinstance(element, NetworkElementPlate):
                        c1 = element.loc - element.w_cap*element.w/2 \
                             - element.h_cap*element.h/2
                        c2 = c1 + element.w_cap * element.w
                        c3 = c2 + element.h_cap * element.h
                        c4 = c3 - element.w_cap * element.w
                        X = [c1[0], c2[0], c3[0], c4[0], c1[0]]
                        Y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
                        Z = [c1[2], c2[2], c3[2], c4[2], c1[2]]
                        ax.plot(X, Y, Z, **style, linewidth=eplot_linewidth)  # plot line
        
        
