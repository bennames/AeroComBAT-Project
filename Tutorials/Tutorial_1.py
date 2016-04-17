# =============================================================================
# AEROCOMBAT TUTORIAL 1 - MATERIAL lIBRARY AND CLASSICAL LAMINATION THEORY
# =============================================================================

# IMPORT SYSTEM PACKAGES
# ======================
import sys
import os
# Append the root to the system path
sys.path.append(os.path.abspath('..'))

# IMPORT AEROCOMBAT CLASSES
# =========================
from AeroComBAT.Structures import MaterialLib, Laminate
from AeroComBAT.Utilities import RotationHelper
from AeroComBAT.tabulate import tabulate
import numpy as np


# MATERIAL lIBRARY VALIDATION
# ===========================
# Generate Empty Material Library
matlib = MaterialLib()
# Add a graphite orthotropic material
matlib.addMat(1, 'Graphite-Polymer Composite ortho', 'ortho', \
                [155.0e9, 12.1e9, 12.1e9, .458, .248, .248, 3.2e9, 4.4e9,\
                4.4e9, 1.7e3], .15e-3)
# Add a graphite transversely isotropic material
matlib.addMat(2, 'Graphite-Polymer Composite', 'trans_iso', \
                [155.0e9, 12.1e9, .458, .248, 4.4e9, 1.7e3], .15e-3)
# Add a glass transversely isotropic material
matlib.addMat(3, 'Glass-Polymer Composite', 'trans_iso', \
                [50.0e9, 15.2e9, .428, .254, 4.7e9, 1.2e3], .15e-3)
# Add a T300 transversely isotropic material
matlib.addMat(4, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3)
# Add a aluminum isotropic material
matlib.addMat(5, 'AL-2050', 'iso',[75.8, 0.33, 2.7e3], .15e-3)
# Add a rotated T300 transversely isotropic material
matlib.addMat(6, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3,th = [0.,45.,0.])
# Print a summary of the mat
matlib.printSummary()
# Get the material associated with MID 1
mat1 = matlib.getMat(1)
# Get the compliance matrix of the material mat1
Smat1 = mat1.Smat
# Get the stiffness matrix of the material mat1 and round for accuracy
Cmat1 = np.around(mat1.Cmat/1e9,decimals=2)
Cmat1[0,0] = np.around(Cmat1[0,0])
# The following matricies are the correct compliance and stiffness matricies
Smat1Test = np.array([[6.45e-12,-1.6e-12,-1.6e-12,0.,0.,0.],\
                      [-1.6e-12,82.6e-12,-37.9e-12,0.,0.,0.],\
                      [-1.6e-12,-37.9e-12,82.6e-12,0.,0.,0.],\
                      [0.,0.,0.,312e-12,0.,0.],\
                      [0.,0.,0.,0.,227e-12,0.],\
                      [0.,0.,0.,0.,0.,227e-12]])
Cmat1Test = np.array([[158e9,5.64e9,5.64e9,0.,0.,0.],\
                      [5.64e9,15.51e9,7.21e9,0.,0.,0.],\
                      [5.64e9,7.21e9,15.51e9,0.,0.,0.],\
                      [0.,0.,0.,3.2e9,0.,0.],\
                      [0.,0.,0.,0.,4.4e9,0.],\
                      [0.,0.,0.,0.,0.,4.4e9]])/1e9
# Check to make sure the calculated values are correct
np.testing.assert_array_almost_equal(Smat1,Smat1Test,decimal=12)
np.testing.assert_array_almost_equal(Cmat1,Cmat1Test,decimal=12)


# MATERIAL PROPERTY ROTATION HELPER VALIDATION
# ============================================
# Create a rotation helper object
rh = RotationHelper()
# Create an array of x-y-z rotations
th = [0.,45.,0.]
# Initialize a stiffness matrix
C = np.array([[1.8403e11,5.4101e9,5.4101e9,0.,0.,0.],\
              [5.4101e9,1.31931e10,6.12866e9,0.,0.,0.],\
              [5.4101e9,6.12866e9,1.31931e10,0.,0.,0.],\
              [0.,0.,0.,5.21455e9,0.,0.],\
              [0.,0.,0.,0.,7.17e9,0.],\
              [0.,0.,0.,0.,0.,7.17e9]])
# Convert it into a compliance matrix
S = np.linalg.inv(C)
# Rotate the compliance matrix
Sp = rh.transformCompl(S,th,xsect=True)
# Convert it back to a stiffness matrix
Cp = np.linalg.inv(Sp)
print('The rotated stiffness matrix:')
print(tabulate(np.around(Cp-C,decimals=3),tablefmt="fancy_grid"))

# =============================================================================
# CLT VALIDATION
# =============================================================================
# Initialize the number of plies per each orientation
n_i = [1,1,1,1]
# Initialize the materials to be used at each orientation
m_i = [4,4,4,4]
# Initialize the angle orientations for the plies
th = [30,-30,0,45]
# Create a laminate with default orientations (for 4 orientations, this will
# default to th_defalt = [0,45,90,-45])
lam1 = Laminate(n_i,m_i,matlib)
# Print a summary of laminate 1
print('Laminate 1 summary:')
lam1.printSummary(decimals=3)
# Create a laminate with default orientations (for more or less than 4
# orientations, th_default = [0]*len(n_i))
lam2 = Laminate(n_i+n_i,m_i+m_i,matlib)
# Print summary of laminate 2
print('Laminate 2 summary:')
lam2.printSummary(decimals=3)
# Create a laminate using the above rotation orientations
lam3 = Laminate(n_i,m_i,matlib,th=th)
# Print Summary of laminate 3
print('Laminate 3 summary:')
lam3.printSummary(decimals=3)