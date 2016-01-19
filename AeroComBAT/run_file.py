# =============================================================================
# RUNFILE
# =============================================================================
import os
import sys
import time

# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================
from Structures import MaterialLib
from Opt import WingOpt

# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
from scipy.optimize import minimize
import numpy as np

# =============================================================================
# DEFINE MATERIAL LIBRARY
# =============================================================================
# Generate Empty Material Library
mat_lib = MaterialLib()
# Add materials to material library. Repeat as necessary
mat_lib.addMat(MID, mat_name, mat_type, mat_constants, mat_t)

# =============================================================================
# CREATE WING OBJECTS
# =============================================================================
# Wing object to be optimized
wing = WingOpt()

# =============================================================================
# OBJECTIVE FUNCTION DEFINITION
# =============================================================================


def wingFunc(x, out='objective', con=0, mat_lib=mat_lib):
    # INITIALIZE CONSTANTS:
    # Free-stream air density
    rho = 1.225    # NOTE: units kg/m^3
    # Free-streem mach number
    M = 0.0    # NOTE, Mach number is for compressebility effects
    # Initialize airfoil profile for zero-lift angle plus cross-section profile
    airfoil = 'NACA0012'
    # Gravitational acceleration:
    g = 9.81    # NOTE: units m/s^2
    # Mass of system minus wings:
    mass = 100    # NOTE: units in kg
    # PARSE VARIABLES:
    # The free stream Air Speed
    V = x[0]
    # The Wing Span
    b_s = x[1]
    # The root chord
    croot = x[2]
    # The tip chord
    ctip = x[3]
    # Non-dimensional starting point of spar
    x0_spar = x[4]
    # Non-dimensional ending point of spar
    x1_spar = x[5]
    # Verify that the provided number of variables is correct
    if not (len(x[6:]))/33 % 33. == 0:
        raise ValueError('The number of panels, ply thicknesses and ply ' +
                         'materials must be perfectly divisible by 3. Check' +
                         ' your variable inputs.')
    # Determine the number of panels (section between ribs)
    n_pan = (len(x[6:]))/33.
    # Rib spanwise 
    Y_rib = x[6:6 + n_pan]
    # Ply thickness vector for a given in-plane fiber rotation
    t_i = x[6+n_pan:6+17*n_pan]
    # Material ID for each ply with a given in-plane fiber rotation
    m_i = x[6+17*n_pan:]
    # Convert "analog" ply thickness for number of plies
    n_ply = np.zeros(len(t_i))
    for i in range(0, len(t_i)):
        t_ply = mat_lib.get_t(m_i[i])
        n_ply[i] = int(round(t_i[i]/t_ply))
    # Create a wing object
    wing.instantiate(b_s, croot, ctip, x0_spar, x1_spar, Y_rib, n_ply, m_i, mat_lib)
    # Add all trim conditions
    wing.addTrim(range(-5, 5))
    # Expected input is a vector. Could be a single value if only one trim
    #  angle interests theuser.
    # CONDUCT STATIC AEROELASTIC ANALYSIS
    wing.staticAero(V, rho, M)
    # CONDUCT FLUTTER ANALYSIS
    wing.flutter()
    # Logic gates to determine what to output
    if out == 'objective':
        return wing.obj
    elif out == 'constraint':
        con = wing.con
        # CONSTRAINT 1: Wing generates enough lift
        if con == 0:
            # con[0] = wing.lift-(mass/2+wing.mass)*g
            return con[0]
        # CONSTRAINT 2: Margin of Safety is greater than zero for all plies
        elif con == 1:
            # con[1] = wing.minMargin()
            return con[1]
        # CONSTRAINT 3: Laminate is not thicker than _______
        elif con == 2:
            # con[2] = wing.maxLamtRatio()
            return con[2]
    else:
        print('No output requested...')
# =============================================================================
# INITIALIZE INITIAL INPUT
# =============================================================================
# Free-stream airspeed
V = 20.   # m/s
# Wing span
b_s = 2    # m
# Root chord
croot = .2    # m
# Tip chord
ctip = .2    # m
# Non-dimensional beam starting point
x0_spar = 0.15
# Non-dimensional beam ending point
x1_spar = 0.35
# Number of ribs
n_ribs = 10
# Y-coordinates of the ribs
Y_rib = np.linspace(0, b_s, n_ribs)
# Vector of ply thicknesses
t_i = []
# Vector of ply material ID's
m_i = []
# Composed initial vector
x0 = [V, b_s, croot, ctip, Y_rib, t_i, m_i]

# =============================================================================
# INITIALIZE VARIABLE BOUNDS
# =============================================================================
bnds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x: wingFunc(x,out='constraint', con=0)},
        {'type': 'ineq', 'fun': lambda x: wingFunc(x,out='constraint', con=1)},
        {'type': 'ineq', 'fun': lambda x: wingFunc(x,out='constraint', con=2)})

# =============================================================================
# RUN OPTIMIZATION
# =============================================================================
res = minimize(wingFunc, x0, method='SLSQP', bounds=bnds, constraints=cons)
