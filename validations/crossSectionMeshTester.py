# =============================================================================
# AEROCOMBAT CROSS-SECTION TESTING SCRIPT
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib, Laminate, XSect
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np

# Material properties from Hodges Cross-section Paper
# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.34,.3,.87e6,0.1],0.005)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.1],0.005)
matLib.addMat(3,'AL','iso',[71.7e9,.33,2810],.005)

# Box Configuration 2
c2 = 0.53
xdim2 = [-.953/(c2*2),.953/(c2*2)]
af2 = Airfoil(c2,name='box')

# Force Vector for Warping Checks:
force = np.array([1000.,0.,0.,0.,0.,0.])


# B1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [15]_6)
n_i_B1 = [6]
m_i_B1 = [1]
th_B1 = [-15]
lam1_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam2_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam3_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam4_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
laminates_B1 = [lam1_B1,lam2_B1,lam3_B1,lam4_B1]
xsect_B1 = XSect(1,af2,xdim2,laminates_B1,matLib,typeXSect='rectBox',meshSize=1)
xsect_B1.xSectionAnalysis()
xsect_B1.plotRigid(mesh=True)
xsect_B1.printSummary(stiffMat=True)
xsect_B1.calcWarpEffects(force=force)
xsect_B1.plotWarped(figName='Validation Case B1',warpScale=10,contLim=[0,500000])
K_b1 = xsect_B1.K_raw
Lam1EIDmesh = lam1_B1.EIDmesh
Lam2EIDmesh = lam2_B1.EIDmesh
Lam3EIDmesh = lam3_B1.EIDmesh
Lam4EIDmesh = lam4_B1.EIDmesh

# Layup 1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [0]_6)
xdim2 = [-.95300001/(c2*2),.95300001/(c2*2)]
n_i_Lay1 = [6]
m_i_Lay1 = [2]
th_Lay1 = [0.]
lam1_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam2_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam3_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam4_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
laminates_Lay1 = [lam1_Lay1,lam2_Lay1,lam3_Lay1,lam4_Lay1]
xsect_Lay1 = XSect(2,af2,xdim2,laminates_Lay1,matLib,typeXSect='rectBox',meshSize=1)
xsect_Lay1.xSectionAnalysis()
xsect_Lay1.printSummary(stiffMat=True)
xsect_Lay1.calcWarpEffects(force=force)
xsect_Lay1.plotWarped(figName='Validation Case L1',warpScale=10,contLim=[0,500000])
KLay1 = xsect_Lay1.K_raw

# Layup 2 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_Lay2 = [1,1,1,1,1,1]
m_i_Lay2 = [2,2,2,2,2,2]
th_Lay2 = [-30,0,-30,0,-30,0]
lam1_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam2_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam3_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam4_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
laminates_Lay2 = [lam1_Lay2,lam2_Lay2,lam3_Lay2,lam4_Lay2]
xsect_Lay2 = XSect(3,af2,xdim2,laminates_Lay2,matLib,typeXSect='rectBox',meshSize=1)
xsect_Lay2.xSectionAnalysis(ref_ax='origin')
xsect_Lay2.printSummary(stiffMat=True)
xsect_Lay2.calcWarpEffects(force=force)
xsect_Lay2.plotWarped(figName='Validation Case L2',warpScale=10,contLim=[0,500000])
KLay2 = xsect_Lay2.K_raw

# Layup 3 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_1 = [1,1,1,1,1,1]
m_i_1 = [2,2,2,2,2,2]
th_1 = [-15,-15,-15,-15,-15,-15]
lam1 = Laminate(n_i_1, m_i_1, matLib, th=th_1)
n_i_2 = [1,1,1,1,1,1]
m_i_2 = [2,2,2,2,2,2]
th_2 = [15,-15,15,-15,15,-15]
lam2 = Laminate(n_i_2, m_i_2, matLib, th=th_2)
n_i_3 = [1,1,1,1,1,1]
m_i_3 = [2,2,2,2,2,2]
th_3 = [15,15,15,15,15,15]
lam3 = Laminate(n_i_3, m_i_3, matLib, th=th_3)
n_i_4 = [1,1,1,1,1,1]
m_i_4 = [2,2,2,2,2,2]
th_4 = [-15,15,-15,15,-15,15]
lam4 = Laminate(n_i_4, m_i_4, matLib, th=th_4)
laminates_Lay3 = [lam1,lam2,lam3,lam4]
xsect_Lay3 = XSect(4,af2,xdim2,laminates_Lay3,matLib,typeXSect='rectBox',meshSize=1)
xsect_Lay3.xSectionAnalysis(ref_ax='origin')
xsect_Lay3.printSummary(stiffMat=True)
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Validation Case L3',warpScale=10,contLim=[0,500000])
KLay3 = xsect_Lay3.K_raw


def reorder(Q):
    scram_vec = [0,1,5,2,4,3]
    newMat = np.zeros((6,6))
    for i in range(0,np.size(Q,axis=0)):
        for j in range(0,np.size(Q,axis=1)):
            newMat[i,j] = Q[scram_vec[i],scram_vec[j]]
    return newMat

Q1 = xsect_Lay3.elemDict[0].Q
Q3 = xsect_Lay3.elemDict[954].Q
Q2_p15 = xsect_Lay3.elemDict[617].Q
Q2_m15 = xsect_Lay3.elemDict[616].Q
Q4_p15 = xsect_Lay3.elemDict[1567].Q
Q4_m15 = xsect_Lay3.elemDict[1566].Q

Q1_Nast = reorder(Q1)
Q3_Nast = reorder(Q3)
Q2_p15_Nast = reorder(Q2_p15)
Q2_m15_Nast = reorder(Q2_m15)
Q4_p15_Nast = reorder(Q4_p15)
Q4_m15_Nast = reorder(Q4_m15)

