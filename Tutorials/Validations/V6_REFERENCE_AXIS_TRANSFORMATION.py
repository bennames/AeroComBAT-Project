# =============================================================================
# HEPHAESTUS VALIDATION 6 - CROSS-SECTION REFERENCE AXIS TRANSFORMATION
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..'))

from Structures import MaterialLib, Laminate, XSect, TBeam, SuperBeam
from AircraftParts import Airfoil
import numpy as np
from tabulate import tabulate

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.34,.34,.87e6,0.],0.004826)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.],.005)

# Box Configuration 2
c2 = 0.53
xdim2 = [-0.8990566037735849,0.8990566037735849]
af2 = Airfoil(c2,name='box')
xdim2_off = [0.,1.7981132075471697]


'''
# B1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [15]_6)
n_i_B1 = [6]
m_i_B1 = [2]
th_B1 = [-15]
lam1_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam2_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam3_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam4_B1 = Laminate(n_i_B1, m_i_B1, matLib, th=th_B1)
lam1_B1.printPlies()
laminates_B1 = [lam1_B1,lam2_B1,lam3_B1,lam4_B1]
xsect_B1 = XSect(af2,xdim2,laminates_B1,matLib,typeXsect='box',meshSize=2)
xsect_B1.xSectionAnalysis()
xsect_B1.printStiffMat()
strn = np.array([0.,0.,0.,0.,0.,1.0])
xsect_B1.strn2dspl(strn,figName='Validation Case B1',contour_Total_T=True)
'''

# Layup 1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [0]_6)
n_i_Lay1 = [6]
m_i_Lay1 = [2]
th_Lay1 = [0]
lam1_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam2_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam3_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam4_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam1_Lay1.printPlies()
laminates_Lay1 = [lam1_Lay1,lam2_Lay1,lam3_Lay1,lam4_Lay1]
xsect_Lay1 = XSect(af2,xdim2,laminates_Lay1,matLib,typeXsect='box',meshSize=2)
xsect_Lay1.xSectionAnalysis()
xsect_Lay1.printStiffMat()
#xsect_Lay1.strn2dspl(strn,figName='Validation Case Layup 1',contour_Total_T=True)

xsect_Lay1_off = XSect(af2,xdim2_off,laminates_Lay1,matLib,typeXsect='box',meshSize=2)
xsect_Lay1_off.xSectionAnalysis()

Ktmp = xsect_Lay1_off.K

print('\n\nThe cross-section stiffness matrix is:')
print(tabulate(np.around(Ktmp,decimals=2),tablefmt="fancy_grid"))

xsc,ysc = xsect_Lay1_off.genXYSC()
xsc = -xsc
ysc = -ysc
T1 = np.array([[1.,0.,0.,0.,0.,-ysc],[0.,1.,0.,0.,0.,xsc],[0.,0.,1.,ysc,-xsc,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]])
T2 = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.],[0.,0.,-ysc,1.,0.,0.],[0.,0.,xsc,0.,1.,0.],[ysc,-xsc,0.,0.,0.,1.]])
Ktrans = np.dot(np.linalg.inv(T2),np.dot(Ktmp,T1))
print('\n\nThe cross-section stiffness matrix is:')
print(tabulate(np.around(Ktrans,decimals=2),tablefmt="fancy_grid"))


X_Lay1 = xsect_Lay1.X
Xtmp = xsect_Lay1_off.X
diff = (X_Lay1-Xtmp)/X_Lay1*100


'''
# Layup 2 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_Lay2 = [1,1,1,1,1,1]
m_i_Lay2 = [2,2,2,2,2,2]
th_Lay2 = [-30,0,-30,0,-30,0]
lam1_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam2_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam3_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam4_Lay2 = Laminate(n_i_Lay2, m_i_Lay2, matLib, th=th_Lay2)
lam1_Lay2.printPlies()
laminates_Lay2 = [lam1_Lay2,lam2_Lay2,lam3_Lay2,lam4_Lay2]
xsect_Lay2 = XSect(af2,xdim2,laminates_Lay2,matLib,typeXsect='box',meshSize=2)
xsect_Lay2.xSectionAnalysis()
xsect_Lay2.printStiffMat()
xsect_Lay2.strn2dspl(strn,figName='Validation Case Layup 2',contour_Total_T=True)


# Layup 2 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
n_i_1 = [1,1,1,1,1,1]
m_i_1 = [1,1,1,1,1,1]
th_1 = [-15,-15,-15,-15,-15,-15]
lam1 = Laminate(n_i_1, m_i_1, matLib, th=th_1)
n_i_2 = [1,1,1,1,1,1]
m_i_2 = [1,1,1,1,1,1]
th_2 = [-15,15,-15,15,-15,15]
lam2 = Laminate(n_i_2, m_i_2, matLib, th=th_2)
n_i_3 = [1,1,1,1,1,1]
m_i_3 = [1,1,1,1,1,1]
th_3 = [15,15,15,15,15,15]
lam3 = Laminate(n_i_3, m_i_3, matLib, th=th_3)
n_i_4 = [1,1,1,1,1,1]
m_i_4 = [1,1,1,1,1,1]
th_4 = [-15,15,-15,15,-15,15]
lam4 = Laminate(n_i_4, m_i_4, matLib, th=th_4)
lam1.printPlies()
lam2.printPlies()
lam3.printPlies()
lam4.printPlies()
laminates_Lay3 = [lam1,lam2,lam3,lam4]
xsect_Lay3 = XSect(af2,xdim2,laminates_Lay3,matLib,typeXsect='box',meshSize=2)
xsect_Lay3.xSectionAnalysis()
xsect_Lay3.printStiffMat()
xsect_Lay3.strn2dspl(strn,figName='Validation Case Layup 3',contour_Total_T=True)
'''

