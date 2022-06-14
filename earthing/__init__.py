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

from math import *

import numpy as np
from numpy.linalg import inv, norm, solve

import matplotlib.pyplot as plt
from matplotlib import path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


## Resistance functions
    
def resistance_plate(rho, A, h):
    """ As per ENA EREC S34
    
        rho: resistivity (ohm-m)
        A: plate area (m^2)
        h = burrial depth
    """
    r = sqrt(A/pi)
    return rho/(8*r)*(1 + r/(2.5*h+r))

def resistance_pipe(rho, L, d):
    """ As per IEEE 80
    
        rho: resistivity (ohm-m)
        L: length of pipe (m)
        d: dia of pipe (m)
    """
    return rho/(2*pi*L)*(log(8*L/d) - 1)

def resistance_strip(rho, L, w, t):
    """ Resistance calculation as per IS-3043
    
        rho: resistivity (ohm-m)
        L: length of strip (m)
        w: depth of burial (m)
        t: thickness of strip/ 2 x dia for pipe  (m)
    """
    return rho/(2*pi*L)*log(2*L**2/(w*t))
        
def resistance_grid(rho, A, L, h=0.5):
    """ Return grid resistance as per IEEE 80
    
        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L: total buried length of conductors (m)
        h: depth of grid (m)
    """
    return rho*(1/L + 1/sqrt(20*A)*(1 + 1/(1+h*sqrt(20/A))))

def resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w):
    """ Return grid resistance with earth rods as per ENA EREC S34
    
        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L_E: length of horizontal electrode (m)
        L_R: length of vertical rod electrode (m)
        N: total number of rods
        s: seperation between rods
        d: diameter of rod electrode
        w: width of tape
    """
    data = np.array([[0,0], [4,2.6], [8,4.3], [12,5.3], 
                     [16,6], [20,6.5], [24,6.8], [40,8], [64,9]])
    fit = np.polyfit(data[:,0], data[:,1], 5)
    poly = np.poly1d(fit)
    k = poly(N)  # Use polynomial fit to get k

    r = sqrt(A/pi)
    R1 = rho/(4*r) + rho/L_E
    
    R_R = rho/(2*pi*L_R)*(log(8*L_R/d)-1)
    alpha = rho/(2*pi*R_R*S)
    R2 = R_R*(1+k*alpha)/N
    
    b = w/pi
    R12 = R1 - rho/(pi*L_E)*(log(L_R/b)-1)
    
    R_E = (R1*R2 - R12**2)/(R1 + R2 - 2*R12)
    return R_E

def resistance_parallel(*res):
    """ Return effective of parallel resistances """
    R = 0
    for r in res:
        R += r**-1
    return R**-1


## Touch & step potential fucntions

def e_step_70(rho, rho_s, h_s, t_s):
    """ Step potential calculation as per IEEE 80
    
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    C_s = 1 - 0.09*(1-rho/rho_s)/(2*h_s+0.09)
    E_step_70 = (1000 + 6*C_s*rho_s)*0.157/sqrt(t_s)
    return E_step_70
    
def e_touch_70(rho, rho_s, h_s, t_s):
    """ Touch potential calculation as per IEEE 80
    
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    C_s = 1 - 0.09*(1-rho/rho_s)/(2*h_s+0.09)
    E_touch_70 = (1000 + 1.5*C_s*rho_s)*0.157/sqrt(t_s)
    return E_touch_70
    
def e_mesh_step_grid(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig):
    """ Touch potential calculation as per IEEE 80
        
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
        Return:
            Em: mesh voltage (V)
            Es: step voltage (V)
    """
    
    # Calculate geometric parameters
    A = Lx*Ly  # Area of grid
    Lc = Lx*Nx + Ly*Ny  # Total length of horizontal conductor in the grid (m)
    L_R = Lr*Nr  # Total length of all earth rods (m)
    Lp = (Lx + Ly)*2  # Length of perimeter conductor (m)
    D = ( (Lx/(Ny-1)) + (Ly/(Nx-1)) )/2  # Spacing between parallel conductors 
                                         # in the mesh (m)
    h0 = 1  # Grid reference depth (m)
    
    # Grid shape components
    n_a = 2*Lc/Lp
    n_b = sqrt(Lp/(4*sqrt(A)))
    n_c = (Lx*Ly/A)**(0.7*A/(Lx*Ly))
    n_d = 1
    n = n_a*n_b*n_c*n_d
    
    Kh = sqrt(1+h/h0)
    if Nr:
        Kii = 1
    else:
        Kii = 1/(2*n)**(2/n)
    
    Ki = 0.644 + 0.148*n
    Km = 1/(2*pi)*( log(D**2/(16*h*d) + (D+2*h)**2/(8*D*d) - h/(4*d))  \
                    + Kii/Kh*log(8/(pi*(2*n-1))) )
    Ks = 1/(pi)*( 1/(2*h) + 1/(D+h) + 1/D*(1-0.5**(n-2)) )
    
    Em = rho*Km*Ki*Ig/( Lc + (1.55 + 1.22*(Lr/sqrt(Lx**2+Ly**2)))*L_R )
    Es = rho*Ks*Ki*Ig/(0.75*Lc + 0.85*L_R)
    
    return Em, Es


## Fault current functions

def fault_current(Un, c, Z):
    """ Fault current calculation as per IEC-60909
    
        Un: Nominal line-line voltage (V)
        c: Voltage factor
        Z: Complex impedence
    """
    return c*Un/(sqrt(3)*abs(Z))

def max_current_density(rho, t):
    """ Electrode maximum current density calculation as per IS-3043
    
        rho: resistivity (ohm-m)
        t: Loading time
        return (A/m2)
    """
    return (7.57e3/sqrt(rho*t))

def current_ratio(rho, R, C, Un, A, L):
    """ Current ratio by C factor method as per ENA EREC S34
    
        rho: resistivity (ohm-m)
        R: combined sorce and destination resistance
        C: C factor
        Un: System voltage (kV)
        A: Cross sectional area (mm2)
        L: Cable length (km)
    """
    return C/(A+9*Un) / sqrt((C/(A+9*Un) + R/L)**2 + 0.6*(rho/A*Un)**0.1)
    

## CALCULATION ELEMENT CLASSES


class DescreteElement:
    """Base class for all unit elements"""
    
    def __init__(self, loc, rho):
        self.loc = loc
        self.rho = rho
        
    def pot_coeff_self(self, loc):
        """ Function to compute self potencial coefficient"""
        pass
    
    def pot_coeff(self, loc, mirror=False):
        """ Function to compute mutual potencial coefficient"""
        pass

    
class DescreteElementCylindrical(DescreteElement):
    """ 1D approximation for cylindrical elememt
        Uses radial current distribution upto critical distance and 
        spherical distribution beyond critical distance
    """
    
    def __init__(self, loc, rho, radius, length):
        DescreteElement.__init__(self, loc, rho)
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


class DescreteElementPlate(DescreteElement):
    """ Circular plate elememt
        Uses equations from the paper:
        Ground impedance of cylindrical metal plate buried in homogeneous earth
        - Slavko Vujević, Zdenko Balaž and Dino Lovrić
    """
    
    def __init__(self, loc, rho, area, normal):
        DescreteElement.__init__(self, loc, rho)
        self.area = area
        self.normal = normal
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


## NETWORK ELEMENT CLASSES


class NetworkElement:
    """Base class for all earth electrod primitives"""
    
    def __init__(self, loc, rho):
        self.loc = np.array(loc)
        self.rho = rho
        self.elements = []
        
    def descretise(self):
        self.elements = []
        
    def get_descrete_elements(self):
        return self.elements
        
        
class NetworkElementPlate(NetworkElement):
    
    def __init__(self, loc, rho, w, h, n_cap=[1,0,0], h_cap=[0,0,1]):
        NetworkElement.__init__(self, loc, rho)
        self.w = w
        self.h = h
        self.n_cap = np.array(n_cap) / norm(n_cap)
        self.h_cap = np.array(h_cap) / norm(h_cap)
        self.w_cap = np.cross(self.h_cap, self.n_cap)
        
    def descretise(self, size):
        NetworkElement.descretise(self)
        nw = int(self.w/size/2)
        nh = int(self.h/size/2)
        delta_w = self.w/(2*nw) * self.w_cap
        delta_h = self.h/(2*nh) * self.h_cap
        area = self.w/nw * self.h/nh / 4
        for i in range(-nw, nw):
            for j in range(-nh, nh):
                loc = (self.loc + delta_w/2 + delta_h/2) + delta_w*i + delta_h*j
                element = DescreteElementPlate(loc, self.rho, area, self.n_cap)
                self.elements.append(element)


class NetworkElementStrip(NetworkElement):
    
    def __init__(self, loc, rho, w, loc_end):
        NetworkElement.__init__(self, loc, rho)
        self.w = w
        self.loc_end = np.array(loc_end)
        
    def descretise(self, size):
        NetworkElement.descretise(self)
        length = norm(self.loc - self.loc_end)
        n = int(length/size)
        delta = length/n
        delta_vec = (self.loc_end - self.loc)/length*delta
        n_cap = [0,0,1]
        area = self.w * norm(delta_vec)
        for i in range(0, n):
            loc = (self.loc + delta_vec/2) + delta_vec * i
            element = DescreteElementPlate(loc, self.rho, area, n_cap)
            self.elements.append(element)
            
    
class NetworkElementPipe(NetworkElement):
    
    def __init__(self, loc, rho, radius, loc_end):
        NetworkElement.__init__(self, loc, rho)
        self.radius = radius
        self.loc_end = np.array(loc_end)
        
    def descretise(self, size):
        NetworkElement.descretise(self)
        
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
            element = DescreteElementPlate(loc, self.rho, area, n_cap)
            self.elements.append(element)
            
            
## NETWORK CLASS
        
    
class Network:
    """Class for network definition and solving"""
    
    def __init__(self, rho):
        self.rho = rho
        self.elements = []
        self.descrete_elements = []
        self.A = None
        self.B = None
        self.X = None
        self.I = None
        self.XX = None
        self.YY = None
        self.V = None
        
    def add_strip(self, loc_start, loc_end, w):
        """ Add strip element
        
            loc_start: Start coordinates of strip
            loc_end: End coordinates of strip
            w: Strip width
        """
        element = NetworkElementStrip(np.array(loc_start), 
                                      self.rho, w, np.array(loc_end))
        self.elements.append(element)
        
    def add_rod(self, loc, radius, length):
        """ Add a vertical rod/ pipe element
        
            loc: Coordinates of pipe (z coordinate defines the rod top)
            radius: Rod radius
            length: Rod length
        """
        loc_start = np.array(loc)
        loc_end = np.array(loc) + np.array([0,0,-length])
        element = NetworkElementPipe(loc_start, self.rho, radius, loc_end)
        self.elements.append(element)
    
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
            loc_start = np.array(loc) + np.array([i*Lx/(Ny-1), 0, -2*w])
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
        self.elements.append(element)
        
    def mesh_geometry(self, desc_size=0.25):
        """Descretise geometry"""
        
        if not self.elements:
            raise Exception('No problem geometry found')
            
        self.descrete_elements = []
        
        # Descretise Geometry
        for element in self.elements:
            element.descretise(desc_size)
            descrete_elements = element.get_descrete_elements()
            self.descrete_elements += descrete_elements
            
    def generate_model(self, desc_size=0.25):
        """Formulate system of equations"""
        
        if not self.elements:
            raise Exception('No problem geometry found')
        
        self.mesh_geometry(desc_size)
                
        # Form matrices
        n = len(self.descrete_elements)
        self.A = np.zeros((n+1, n+1))
        self.B = np.zeros((n+1, 1))
        self.B[n, 0] = 1  # current injection 1A
        
        # AX = B ; the system of equation to be solved (n+1)x(n+1)
        #
        # sum{(Pij+Pij')*Ii} - V = 0  -- n equations
        # I1 + I2 + ... + 0*V = 1A
        #
        # X = [I1 I2 ... Ii ... In V]

        # Computation of elements of A
        for i, element_cur in enumerate(self.descrete_elements):
            for j, element_rem in enumerate(self.descrete_elements):
                if i == j:  # set diag terms
                    self.A[i,i] = element_cur.pot_coeff_self() \
                                  +  element_cur.pot_coeff(element_cur.loc, 
                                                           mirror=True)
                else:  # set off diag terms
                    self.A[i,j] = element_cur.pot_coeff(element_rem.loc) \
                                  + element_cur.pot_coeff(element_rem.loc, 
                                                          mirror=True) 
        
        # set last column and row
        v = np.ones((1,n))
        self.A[n, :n] = v
        self.A[:n, n] = -v
        
    def generate_model_fast(self, desc_size=0.25):
        """ Formulate system of equations
            Assume circular plate descretisation for fast calculation 
        """
        
        if not self.elements:
            raise Exception('No problem geometry found')
        
        self.mesh_geometry(desc_size)
        
        # AX = B ; the system of equation to be solved (n+1)x(n+1)
        #
        # sum{(Pij+Pij')*Ii} - V = 0  -- n equations
        # I1 + I2 + ... + 0*V = 1A
        #
        # X = [I1 I2 ... Ii ... In V]
                
        # Form matrices
        
        n = len(self.descrete_elements)
        self.A = np.zeros((n+1, n+1))
        self.B = np.zeros((n+1, 1))
        self.B[n, 0] = 1  # current injection 1A
        
        # Form distance array
        
        A = np.zeros(n)
        N = np.zeros(n, dtype='object')
        LOC = np.zeros(n, dtype='object')
        LOCm = np.zeros(n, dtype='object')
        
        for slno, element in enumerate(self.descrete_elements):
            mirror_loc = np.copy(element.loc)
            mirror_loc[2] = -mirror_loc[2]
            LOC[slno] = element.loc
            LOCm[slno] = mirror_loc
            N[slno] = element.normal
            A[slno] = element.radius
        
        LOC2 = np.repeat(LOC[:, np.newaxis], n, axis=1)
        NN = np.repeat(N[:, np.newaxis], n, axis=1)
        AA = np.repeat(A[:, np.newaxis], n, axis=1)
        LOC2m = np.repeat(LOCm[:, np.newaxis], n, axis=1)
        LOC2r = np.repeat(LOC[np.newaxis, :], n, axis=0)
        
        DD = LOC2 - LOC2r  # Distance from current point to remote point
        DDm = LOC2m - LOC2r  # Distance from mirror point to remote point
        
        # Computation of elements of A
        
        e_norm = np.vectorize(norm)

        Z = e_norm(N * DD)
        R = e_norm(DD - (N * DD))
        Zm = e_norm(N * DDm)
        Rm = e_norm(DDm - (N * DDm))
        
        A_ = R**2 + Z**2 - AA**2
        ALPHA = np.sqrt((A_ + np.sqrt(A_**2 + 4 * AA**2 * Z**2))/2)
        COEF = self.rho/(4*pi*AA)*np.arctan(AA/ALPHA)
        A_m = Rm**2 + Zm**2 - AA**2
        ALPHAm = np.sqrt((A_m + np.sqrt(A_m**2 + 4 * AA**2 * Zm**2))/2)
        COEFm = self.rho/(4*pi*AA)*np.arctan(AA/ALPHAm)
        
        v = np.ones((1,n))
        
        self.A[0:n,0:n] = COEF + COEFm
        self.A[n, :n] = v
        self.A[:n, n] = -v
        
    def solve_model(self):
        """Solve system of equations"""
        
        if not self.descrete_elements:
            raise Exception('No model generated')
            
        self.X = solve(self.A, self.B)
        self.I = self.X[:-1,0]
    
    def get_resistance(self):
        """Get overall resistance of grid"""
        
        if self.X is None:
            raise Exception('Model not solved')
            
        n = len(self.descrete_elements)
        res = self.X[n,0]
        return np.round(res, 3)
    
    def gpr(self, Ig):
        """ Calculate ground potential rise of grid
        
            Ig: Grid fault current
        """
        
        if self.X is None:
            raise Exception('Model not solved')
            
        return np.round(Ig * self.get_resistance(), 3)
    
    def get_point_potential(self, loc, Ig=1):
        """ Get potential at a fixed location
        
            loc: Location for calculation
            Ig: Grid fault current
        """
        
        if self.X is None:
            raise Exception('Model not solved')
            
        v = 0
        for slno, element in enumerate(self.descrete_elements):
            coeff = element.pot_coeff(loc) + element.pot_coeff(loc, mirror=True)
            v += coeff * self.I[slno] * Ig
        return v
    
    def solve_surface_potential(self, Ig=1, grid=(20,20), 
                                xlim=(-20, 20), ylim=(-20, 20)):
        """ Solve for surface potentials on a uniform grid
        
            Ig: Grid fault current
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
                V[i, j] = self.get_point_potential(loc, Ig)
        self.XX = X
        self.YY = Y
        self.V = V
        
    def solve_surface_potential_fast(self, Ig=1, grid=(20,20), 
                                     xlim=(-20, 20), ylim=(-20, 20), 
                                     save_results=True):
        """ Solve for surface potentials on a uniform grid
            Assumes sperical potential distribution of elements for fast 
            calculation 
            
            Ig: Grid fault current
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
        
        # Calculate distance array
        nd = len(self.descrete_elements)
        XXX = np.repeat(XX[:, :, np.newaxis], nd, axis=2)
        YYY = np.repeat(YY[:, :, np.newaxis], nd, axis=2)
        Xe = np.zeros(nd)
        Ye = np.zeros(nd)
        Ze = np.zeros(nd)
        for slno, element in enumerate(self.descrete_elements):
            Xe[slno], Ye[slno], Ze[slno] = element.loc
        XXXe = np.tile(Xe, grid[0]*grid[1]).reshape((grid[1], grid[0], nd))
        YYYe = np.tile(Ye, grid[0]*grid[1]).reshape((grid[1], grid[0], nd))
        ZZZe = np.tile(Ze, grid[0]*grid[1]).reshape((grid[1], grid[0], nd))
        DDD = np.sqrt((XXXe-XXX)**2 + (YYYe-YYY)**2 +  ZZZe**2)
        
        # Calculate voltage arrays
        # Symmetry allows doubling of voltage on account of mirror geometry
        III = np.tile(self.I, grid[0]*grid[1]).reshape((grid[1], grid[0], nd))
        VVV = III * Ig * self.rho/(4*pi*DDD) * 2 
        VV = np.sum(VVV, axis=2)
        
        if save_results:
            self.XX = XX
            self.YY = YY
            self.V = VV
        else:
            return VV
        
    def mesh_voltage(self, Ig, polygon_points, mesh_no=10, 
                     plot=False, plot_type='fill', levels=10, grid_spacing=1):
        """ Find mesh voltage in the passed polygon
        
            Parameters:
                Ig: Grid current
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
        V = self.solve_surface_potential_fast(Ig, grid, xlim, ylim, 
                                              save_results=False)
        
        # Prepare mask with polygon
        xx, yy = np.meshgrid(np.linspace(x0, x1, x_samples),
                             np.linspace(y0, y1, y_samples))
        index_array = np.hstack((xx.flatten()[:,np.newaxis], 
                                 yy.flatten()[:,np.newaxis]))
        flags = polygon.contains_points(index_array).reshape((grid[1], grid[0]))
             
        # Find mesh voltage from masked voltage
        V_masked = (self.gpr(Ig) - V) * flags
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
        
    def step_voltage(self, Ig, polygon_points, mesh_no=14, step_size=1, 
                     plot=False, plot_type='fill', levels=10, grid_spacing=1):
        """ Find step voltage in the passed polygon
            Function shifts voltage mesh grids in 8 directions for evaluation
            of step voltage at various points
            
            Parameters:
                Ig: Grid current
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
        y0 = np.min(polygon_array[:,1]) - offset + 1/mesh_no  # Hack to cutoff 
                                                              # transition 
                                                              # from 0 to first 
                                                              # point
        y1 = np.max(polygon_array[:,1]) + offset
        xlim = (x0, x1)
        ylim = (y0, y1)
        x_samples = int((x1-x0)*mesh_no) + 1 + offset*mesh_no*2
        y_samples = int((y1-y0)*mesh_no) + 1 + offset*mesh_no*2
        grid = (x_samples, y_samples)
        V = self.solve_surface_potential_fast(Ig, grid, xlim, ylim, 
                                              save_results=False)
        
        # Prepare mask with polygon
        xx, yy = np.meshgrid(np.linspace(x0, x1, x_samples),
                             np.linspace(y0, y1, y_samples))
        index_array = np.hstack((xx.flatten()[:,np.newaxis], 
                                 yy.flatten()[:,np.newaxis]))
        flags = polygon.contains_points(index_array).reshape((grid[1], grid[0]))
        
        # Compute step voltage array
        directions = np.array([(1,0), (0.707,0.707), (0,1), (-0.707, 0.707)])
        step_array = np.round(directions * mesh_no * step_size / 2)
        V_steps = np.zeros((y_samples, x_samples, directions.shape[0]))
        for slno, shift_values in enumerate(step_array.astype('i')):
            x_shift, yshift = shift_values
            V0 = np.roll(V, (-x_shift, -yshift), axis=(0, 1))
            V1 = np.roll(V, (x_shift, yshift), axis=(0, 1))
            V_steps[:,:,slno] = np.absolute(V1-V0)
            
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
            if self.X is None:
                raise Exception('Model not solved')
            XX = self.XX
            YY = self.YY
            V = self.V
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
        for element in self.elements:
            if isinstance(element, NetworkElementStrip) \
               or isinstance(element, NetworkElementPipe):
                start = element.loc
                end = element.loc_end
                X = [start[0], end[0]]
                Y = [start[1], end[1]]
                ax.plot(X, Y, color='green')  # plot line
            if isinstance(element, NetworkElementPlate):
                c1 = element.loc - element.w_cap*element.w/2 \
                     - element.h_cap*element.h/2
                c2 = c1 + element.w_cap * element.w
                c3 = c2 + element.h_cap * element.h
                c4 = c3 - element.w_cap * element.w
                X = [c1[0], c2[0], c3[0], c4[0], c1[0]]
                Y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
                ax.plot(X, Y, color='green')  # plot line
        
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
            normal: Show descrete element normal vectors
            normal_scale: Descrete element normal vector scale
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
            contour_plot = ax.contourf(self.XX, self.YY, self.V, zdir='z', 
                                       offset=0, cmap="plasma", alpha=0.7)
            cbar = plt.colorbar(contour_plot, pad=0.1)
            # cbar.set_label('Volts', loc='bottom')
        
        if current_distribution:
            # Plot problem geometry with current as weight
            X = []
            Y = []
            Z = []
            for slno, element in enumerate(self.descrete_elements):
                loc = element.loc
                X.append(loc[0])
                Y.append(loc[1])
                Z.append(loc[2])
            scat_plot = ax.scatter(X, Y, Z, c=self.I, s=5, cmap='viridis')
            cbar = plt.colorbar(scat_plot, pad=0.1)
            # cbar.set_label('Amps', loc='bottom')
        else:
            # Plot problem geometry as lines
            for element in self.elements:
                if isinstance(element, NetworkElementStrip) \
                   or isinstance(element, NetworkElementPipe):
                    start = element.loc
                    end = element.loc_end
                    X = [start[0], end[0]]
                    Y = [start[1], end[1]]
                    Z = [start[2], end[2]]
                    ax.plot(X, Y, Z, color='green')  # plot line
                if isinstance(element, NetworkElementPlate):
                    c1 = element.loc - element.w_cap*element.w/2 \
                         - element.h_cap*element.h/2
                    c2 = c1 + element.w_cap * element.w
                    c3 = c2 + element.h_cap * element.h
                    c4 = c3 - element.w_cap * element.w
                    X = [c1[0], c2[0], c3[0], c4[0], c1[0]]
                    Y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
                    Z = [c1[2], c2[2], c3[2], c4[2], c1[2]]
                    ax.plot(X, Y, Z, color='green')  # plot line
        
        # Plot direction vectors for descrete elements
        if normal:
            X = []
            Y = []
            Z = []
            U = []
            V = []
            W = []    
            for desc_element in self.descrete_elements:
                if isinstance(desc_element, DescreteElementPlate):
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
    
    
    
