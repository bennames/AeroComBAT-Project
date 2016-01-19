# =============================================================================
# HEPHAESTUS VALIDATION 2 - MATERIAL lIBRARY AND CLT VALIDATION
# =============================================================================

from Structures import MaterialLib, Laminate
from Utilities import RotationHelper
from tabulate import tabulate
import numpy as np


# =============================================================================
# MATERIAL lIBRARY VALIDATION
# =============================================================================
# Generate Empty Material Library
mat_lib = MaterialLib()
# Add materials to material library. Repeat as necessary
mat_lib.addMat(1, 'Graphite-Polymer Composite', 'trans_iso', \
                [155.0e9, 12.1e9, .458, .248, 4.4e9, 1.7e3], .15e-3)
mat_lib.addMat(2, 'Glass-Polymer Composite', 'trans_iso', \
                [50.0e9, 15.2e9, .428, .254, 4.7e9, 1.2e3], .15e-3)
mat_lib.addMat(3, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3,xsect=True)
mat_lib.addMat(4, 'AL-2050', 'iso', \
                [75.8, 0.33, 2.7e3], .15e-3,xsect=True)
mat3 = mat_lib.matDict[3]
mat3.printSummary(stiffness=True)
mat_lib.addMat(5, 'T300/5208', 'trans_iso', \
                [181.0e9, 10.3e9, .458, .28, 7.17e9, 1.8e3], .15e-3,xsect=True,th = [45.,0.,0.])
mat5 = mat_lib.matDict[5]
mat5.printSummary(stiffness=True)

mat_lib.printMaterialSummary()

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
lam1.printABD(decimals=8)
lam1.printPlies()