# =============================================================================
# HEPHAESTUS VALIDATION 4 - MESHER AND CROSS-SECTIONAL ANALYSIS
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..\..'))

from AeroComBAT.Structures import Node, CQUADX, CQUADX9, MaterialLib, XSect
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np

# HODGES XSECTION VALIDATION

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.3,.34,.87e6,0.00177],0.005)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.00177],0.005)
matLib.addMat(3,'AL','iso',[71.7e9,.33,2810],.005)

n1 = Node(1,[0.,0.,0.])
n2 = Node(2,[.5,0.,0.])
n3 = Node(3,[1.,0.,0.])
n4 = Node(4,[0.,.5,0.])
n5 = Node(5,[.5,.5,0.])
n6 = Node(6,[1.,.5,0.])
n7 = Node(7,[0.,1.,0.])
n8 = Node(8,[.5,1.,0.])
n9 = Node(9,[1.,1.,0.])

nodes = [n1,n2,n3,n4,n5,n6,n7,n8,n9]

elem1 = CQUADX9(1,nodes,3,matLib)

N1 = Node(1,[0.,0.,0.])
N2 = Node(2,[1.,0.,0.])
N3 = Node(3,[1.,1.,0.])
N4 = Node(4,[0.,1.,0.])
nodes1 = [N1,N2,N3,N4]
elem2 = CQUADX(2,nodes1,3,matLib)

b = 1
h = 1
elemX = 24
elemY = elemX
xsect1 = XSect(1,None,[b,h],None,matLib,elemX=elemX,elemY=elemY,typeXSect='solidBox',\
    MID=3,elemOrder=1)
xsect1.xSectionAnalysis()
xsect2 = XSect(1,None,[b,h],None,matLib,elemX=elemX,elemY=elemY,typeXSect='solidBox',\
    MID=3,elemOrder=2)
xsect2.xSectionAnalysis()

E = 71.7e9
nu = .33
G = E/(2*(1+nu))
A = 1
k = 5./6
I = b*h**3/12.
Ktrue = np.array([[G*A*k,0,0,0,0,0],[0,G*A*k,0,0,0,0],[0,0,E*A,0,0,0],\
    [0,0,0,E*I,0,0],[0,0,0,0,E*I,0],[0,0,0,0,0,G*2.25*(b/2.)**4]])
Kapprox1 = xsect1.K
Kapprox2 = xsect2.K