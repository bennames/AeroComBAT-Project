# =============================================================================
# HEPHAESTUS VALIDATION 2 - MATERIAL lIBRARY AND CLT VALIDATION
# =============================================================================

import sys
import os

sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib, Laminate
from AeroComBAT.Utilities import RotationHelper
from AeroComBAT.tabulate import tabulate
import numpy as np



# =============================================================================
# MATERIAL lIBRARY VALIDATION
# =============================================================================
# Generate Empty Material Library
mat_lib = MaterialLib()
# Add materials to material library. Repeat as necessary
mat_lib.addMat(1, 'Graphite-Polymer Composite ortho', 'ortho', \
                [155.0e9, 12.1e9, 12.1e9, .458, .248, .248, 3.2e9, 4.4e9,\
                4.4e9, 1.7e3], .15e-3)
mat_lib.addMat(2, 'Graphite-Polymer Composite', 'trans_iso', \
                [155.0e9, 12.1e9, .458, .248, 4.4e9, 1.7e3], .15e-3)
mat_lib.addMat(3, 'Glass-Polymer Composite', 'trans_iso', \
                [50.0e9, 15.2e9, .428, .254, 4.7e9, 1.2e3], .15e-3)
mat_lib.addMat(4, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3)
mat_lib.addMat(5, 'AL-2050', 'iso', \
                [75.8, 0.33, 2.7e3], .15e-3)
mat_lib.addMat(6, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3,th = [45.,0.,0.])
mat_lib.printSummary()

mat1 = mat_lib.getMat(1)
Smat1 = mat1.Smat
Cmat1 = np.around(mat1.Cmat/1e9)
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
np.testing.assert_array_almost_equal(Smat1,Smat1Test,decimal=12)
#np.testing.assert_array_almost_equal(Cmat1,Cmat1Test,decimal=2)

# =============================================================================
# MATERIAL PROPERTY ROTATION HELPER VALIDATION
# =============================================================================
rh = RotationHelper()
th = [0.,45.,0.]
C = np.array([[1.8403e11,5.4101e9,5.4101e9,0.,0.,0.],\
              [5.4101e9,1.31931e10,6.12866e9,0.,0.,0.],\
              [5.4101e9,6.12866e9,1.31931e10,0.,0.,0.],\
              [0.,0.,0.,5.21455e9,0.,0.],\
              [0.,0.,0.,0.,7.17e9,0.],\
              [0.,0.,0.,0.,0.,7.17e9]])
S = np.linalg.inv(C)
Sp = rh.transformCompl(S,th,xsect=True)
Cp = np.linalg.inv(Sp)
print(tabulate(np.around(Cp,decimals=3),tablefmt="fancy_grid"))

# =============================================================================
# CLT VALIDATION
# =============================================================================
# Make a laminate:
n_i_tmp = [1,1,1,1]
m_i_tmp = [4,4,4,4]
#th = [30,-30,0]
lam1 = Laminate(n_i_tmp,m_i_tmp,mat_lib)
lam2 = Laminate(n_i_tmp,m_i_tmp,mat_lib)
lam3 = Laminate(n_i_tmp,m_i_tmp,mat_lib)
lam4 = Laminate(n_i_tmp,m_i_tmp,mat_lib)
lam1.printSummary(decimals=5)