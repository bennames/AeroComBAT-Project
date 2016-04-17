# =============================================================================
# AEROCOMBAT TUTORIAL 2 - CQUADX AND AIRFOIL
# =============================================================================

# IMPORT SYSTEM PACKAGES
# ======================
import sys
import os

sys.path.append(os.path.abspath('..'))

# IMPORT AEROCOMBAT CLASSES
# =========================
from AeroComBAT.Structures import Node, MaterialLib, CQUADX
from AeroComBAT.Aerodynamics import Airfoil

# IMPORT NUMPY MODULES
# ====================
import numpy as np
import matplotlib.pyplot as plt

# Material Info
mat_lib = MaterialLib()
# Add an aluminum isotropic material
mat_lib.addMat(1, 'AL-2050', 'iso',[75.8, 0.33, 2.7e3], .15e-3)
                

# CQUADX 2D ELEMENT CREATION
# ==========================
# Create a node 1 object
n1 = Node(1,[0.,0.,0.])
# Create a node 2 object
n2 = Node(2,[2.,0.,0.])
# Create a node 3 object
n3 = Node(3,[2.,3.,0.])
# Create a node 4 object
n4 = Node(4,[0.,5.,0.])
# Create a CQUADX element
elem1 = CQUADX(1,[n1,n2,n3,n4],1,mat_lib)
# Print a summary of the element
elem1.printSummary(nodes=True)

# AIRFOIL OUTER MOLD LINE VALIDATION
# ==================================
# Initialize a chord length of 1
c = 1.
# Create an airfoil object with a 'box' profile
af1 = Airfoil(c,name='box')
# Generate a set of non-dimensional x-coordinates
x = np.linspace(-.5,.5,50)
# Create the upper and lower box airfoil curves
xu,yu,xl,yl = af1.points(x)
# Create a matplotlib figure
plt.figure(num=1)
plt.plot(xu,yu)
plt.hold(True)
plt.plot(xl,yl)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel('x coordinate along the airfoil')
plt.ylabel('y coordinate along the airfoil')
plt.title('Box airfoil profile')
plt.hold(False)

# Create a NACA2412 airfoil profile
af2 = Airfoil(c,name='NACA2412')
# Generate a set of non-dimensional x-coordinates
x = np.linspace(0,1.,500)
# Create the upper and lower airfoil curves
xu,yu,xl,yl = af2.points(x)
# Create a matplotlib figure
plt.figure(num=2)
plt.plot(xu,yu)
plt.hold(True)
plt.plot(xl,yl)
plt.hold(False)
plt.axes().set_aspect('equal', 'datalim')