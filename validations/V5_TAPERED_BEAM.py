# =============================================================================
# HEPHAESTUS VALIDATION 5 - TAPERED BEAM STIFFNESS SENSITIVITY
# =============================================================================

# IMPORTS:
from Structures import MaterialLib, Laminate, XSect
from AircraftParts import Airfoil
import numpy as np
import pylab as pl

# HODGES XSECTION VALIDATION

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.34,.34,.87e6,0.],0.004826)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.],.005)

# Box Configuration 2
c = 1.5
xdim = [-0.8990566037735849,0.8990566037735849]
strn = np.array([0.,0.,0.,0.,0.,1.0])

num_data_points = 50
cs = np.linspace(0.1,1,num_data_points)

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

xsectDict = {}

for i in range(len(cs)):
    xsectDict['xsect'+str(i)] = XSect(Airfoil(cs[i],name='box'),xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
    xsectDict['xsect'+str(i)].xSectionAnalysis()
    #xsectDict['xsect'+str(i)].strn2dspl(strn,figName='xsect'+str(i),contour_Total_T=True)
'''
xsect1 = XSect(af1,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect2 = XSect(af2,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect3 = XSect(af3,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect4 = XSect(af4,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect5 = XSect(af5,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect6 = XSect(af6,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect7 = XSect(af7,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect8 = XSect(af8,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect9 = XSect(af9,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)
xsect10 = XSect(af10,xdim,laminates_Lay3,matLib,typeXsect='box',meshSize=4)

xsect1.xSectionAnalysis()
xsect2.xSectionAnalysis()
xsect3.xSectionAnalysis()
xsect4.xSectionAnalysis()
xsect5.xSectionAnalysis()
xsect6.xSectionAnalysis()
xsect7.xSectionAnalysis()
xsect8.xSectionAnalysis()
xsect9.xSectionAnalysis()
xsect10.xSectionAnalysis()'''

#xsects = [xsect1,xsect2,xsect3,xsect4,xsect5,xsect6,xsect7,xsect8,xsect9,xsect10]

#xsect1.strn2dspl(strn,figName='xsect1',contour_Total_T=True)
#xsect10.strn2dspl(strn,figName='xsect10',contour_Total_T=True)

K11 = []
K12 = []
K13 = []
K14 = []
K15 = []
K16 = []
K22 = []
K23 = []
K24 = []
K25 = []
K26 = []
K33 = []
K34 = []
K35 = []
K36 = []
K44 = []
K45 = []
K46 = []
K55 = []
K56 = []
K66 = []

for key, sect in xsectDict.iteritems():
    K11 += [sect.K[0,0]]
    K12 += [sect.K[0,1]]
    K13 += [sect.K[0,2]]
    K14 += [sect.K[0,3]]
    K15 += [sect.K[0,4]]
    K16 += [sect.K[0,5]]
    K22 += [sect.K[1,1]]
    K23 += [sect.K[1,2]]
    K24 += [sect.K[1,3]]
    K25 += [sect.K[1,4]]
    K26 += [sect.K[1,5]]
    K33 += [sect.K[2,2]]
    K34 += [sect.K[2,3]]
    K35 += [sect.K[2,4]]
    K36 += [sect.K[2,5]]
    K44 += [sect.K[3,3]]
    K45 += [sect.K[3,4]]
    K46 += [sect.K[3,5]]
    K55 += [sect.K[4,4]]
    K56 += [sect.K[4,5]]
    K66 += [sect.K[5,5]]
Ks = [K11,K12,K13,K14,K15,K16,K22,K23,K24,K25,K26,K33,K34,K35,K36,K44,K45,K46,K55,K56,K66]

labels = ['K11','K12','K13','K14','K15','K16','K22','K23','K24','K25','K26',\
'K33','K34','K35','K36','K44','K45','K46','K55','K56','K66']

import matplotlib.pyplot as plt

plt.figure(num=4)
plt.hold(True)
plt.axes().set_aspect('equal', 'datalim')
for i in range(0,len(Ks)):
    k = np.array(Ks[i])
    plt.plot(cs,k/max(abs(k)),label=labels[i])
plt.xlim(0,1.)
plt.legend()
plt.grid(True)
plt.title('Normalized Stiffness Coefficients')
plt.xlabel('Percent Chord')
plt.hold(False)
