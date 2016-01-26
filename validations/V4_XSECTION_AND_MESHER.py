# =============================================================================
# HEPHAESTUS VALIDATION 4 - MESHER AND CROSS-SECTIONAL ANALYSIS
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib, Laminate, XSect, TBeam, SuperBeam
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np

# HODGES XSECTION VALIDATION

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.34,.34,.87e6,0.1],0.005)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.1],0.005)

# Box Configuration 2
c2 = 0.53
xdim2 = [-0.8990566037735849,0.8990566037735849]
af2 = Airfoil(c2,name='box')



# B1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [15]_6)
n_i_B1 = [6]
m_i_B1 = [1]
th_B1 = [-15]
lam1_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam2_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam3_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam4_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam1_B1.printSummary()
laminates_B1 = [lam1_B1,lam2_B1,lam3_B1,lam4_B1]
xsect_B1 = XSect(1,af2,xdim2,laminates_B1,matLib,typeXsect='box',meshSize=1)
xsect_B1.xSectionAnalysis()
xsect_B1.printSummary(stiffMat=True)
force = np.array([0.,0.,0.,0.,0.,1e3])
xsect_B1.calcWarpEffects(force=force)
xsect_B1.plotWarped(figName='Validation Case B1',warpScale=10,contLim=[0,500000])


# Layup 1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [0]_6)
n_i_Lay1 = [6]
m_i_Lay1 = [2]
th_Lay1 = [0]
lam1_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam2_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam3_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam4_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam1_Lay1.printSummary()
laminates_Lay1 = [lam1_Lay1,lam2_Lay1,lam3_Lay1,lam4_Lay1]
xsect_Lay1 = XSect(2,af2,xdim2,laminates_Lay1,matLib,typeXsect='box',meshSize=1)
xsect_Lay1.xSectionAnalysis()
xsect_Lay1.printSummary(stiffMat=True)
xsect_Lay1.calcWarpEffects(force=force)
xsect_Lay1.plotWarped(figName='Validation Case L1',warpScale=10,contLim=[0,500000])

# Layup 2 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_Lay2 = [1,1,1,1,1,1]
m_i_Lay2 = [2,2,2,2,2,2]
th_Lay2 = [-30,0,-30,0,-30,0]
lam1_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam2_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam3_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam4_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam1_Lay2.printSummary()
laminates_Lay2 = [lam1_Lay2,lam2_Lay2,lam3_Lay2,lam4_Lay2]
xsect_Lay2 = XSect(3,af2,xdim2,laminates_Lay2,matLib,typeXsect='box',meshSize=1)
xsect_Lay2.xSectionAnalysis()
xsect_Lay2.printSummary(stiffMat=True)
xsect_Lay2.calcWarpEffects(force=force)
xsect_Lay2.plotWarped(figName='Validation Case L2',warpScale=10,contLim=[0,500000])


# Layup 2 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_1 = [1,1,1,1,1,1]
m_i_1 = [2,2,2,2,2,2]
th_1 = [-15,-15,-15,-15,-15,-15]
lam1 = Laminate(n_i_1, m_i_1, matLib, th=th_1)
n_i_2 = [1,1,1,1,1,1]
m_i_2 = [2,2,2,2,2,2]
th_2 = [-15,15,-15,15,-15,15]
lam2 = Laminate(n_i_2, m_i_2, matLib, th=th_2)
n_i_3 = [1,1,1,1,1,1]
m_i_3 = [2,2,2,2,2,2]
th_3 = [15,15,15,15,15,15]
lam3 = Laminate(n_i_3, m_i_3, matLib, th=th_3)
n_i_4 = [1,1,1,1,1,1]
m_i_4 = [2,2,2,2,2,2]
th_4 = [-15,15,-15,15,-15,15]
lam4 = Laminate(n_i_4, m_i_4, matLib, th=th_4)
lam1.printSummary()
lam2.printSummary()
lam3.printSummary()
lam4.printSummary()
laminates_Lay3 = [lam1,lam2,lam3,lam4]
xsect_Lay3 = XSect(4,af2,xdim2,laminates_Lay3,matLib,typeXsect='box',meshSize=1)
xsect_Lay3.xSectionAnalysis()
xsect_Lay3.printSummary(stiffMat=True)
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Validation Case L3',warpScale=10,contLim=[0,500000])


