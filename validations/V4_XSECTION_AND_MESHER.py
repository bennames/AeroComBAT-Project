# =============================================================================
# HEPHAESTUS VALIDATION 4 - MESHER AND CROSS-SECTIONAL ANALYSIS
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib, Laminate, XSect
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np

# HODGES XSECTION VALIDATION

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.3,.34,.87e6,0.00177],0.005)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.00177],0.005)
matLib.addMat(3,'AL','iso',[71.7e9,.33,2810],.005)

# Box Configuration 2
c2 = 0.53
xdim2 = [-0.8990566037735849,0.8990566037735849]
af2 = Airfoil(c2,name='box')
force = np.array([100,100,10000,-400.89,400.89,0.])
#force = np.array([0.,0.,1e4,-8.028e-6,0.,0.])

'''
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
xsect_B1 = XSect(1,af2,xdim2,laminates_B1,matLib,typeXsect='box',meshSize=4)
xsect_B1.xSectionAnalysis()
xsect_B1.printSummary(stiffMat=True)
xsect_B1.calcWarpEffects(force=force)
xsect_B1.plotWarped(figName='Validation Case B1',warpScale=10,contLim=[0,500000])
K = xsect_B1.K_raw


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
xsect_Lay1 = XSect(2,af2,xdim2,laminates_Lay1,matLib,typeXsect='box',meshSize=3)
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
xsect_Lay2 = XSect(3,af2,xdim2,laminates_Lay2,matLib,typeXsect='box',meshSize=3)
xsect_Lay2.xSectionAnalysis(ref_ax='origin')
xsect_Lay2.printSummary(stiffMat=True)
xsect_Lay2.calcWarpEffects(force=force)
xsect_Lay2.plotWarped(figName='Validation Case L2',warpScale=10,contLim=[0,500000])

'''
'''
# Layup 3 Box beam (0.5 x 0.923 in^2 box with laminate schedule [30,0]_3)
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
'''

# AL Box Beam
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
'''
laminates_Lay3 = [lam1,lam2,lam3,lam4]
xsect_Lay3 = XSect(4,af2,xdim2,laminates_Lay3,matLib,typeXsect='box',meshSize=1)
xsect_Lay3.xSectionAnalysis()
K = xsect_Lay3.K_raw

force = np.array([1,0,0,0,0,0.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement Fx',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement Fx',warpScale=2e9,contLim=[0,1e-9])

force = np.array([0,1,0,0,0,0.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement Fy',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement Fy',warpScale=1e9,contLim=[0,1e-9])

force = np.array([0,0,1,0,0,0.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement Fz',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement Fz',warpScale=5e9,contLim=[0,1e-9])

force = np.array([0,0,0,1,0,0.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement Mx',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement Mx',warpScale=1e9,contLim=[0,1e-9])

force = np.array([0,0,0,0,1,0.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement My',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement My',warpScale=1e9,contLim=[0,1e-9])

force = np.array([0,0,0,0,0,1.])
xsect_Lay3.calcWarpEffects(force=force)
xsect_Lay3.plotWarped(figName='Warping Displacement Mz',warpScale=0,contour='none')
xsect_Lay3.plotWarped(figName='Warping Displacement Mz',warpScale=5e8,contLim=[0,1e-9])
'''

lam1.printSummary()
lam2.printSummary()
lam3.printSummary()
lam4.printSummary()
laminates_Lay3 = [lam1,lam2,lam3,lam4]
xsect_Lay3 = XSect(4,af2,xdim2,laminates_Lay3,matLib,typeXSect='rectBox',meshSize=1)
xsect_Lay3.xSectionAnalysis()
xsect_Lay3.printSummary(stiffMat=True)
xsect_Lay3.calcWarpEffects(force=force)

import mayavi.mlab as mlab

xsect_Lay3.plotWarped(figName='Stress sig_11',warpScale=1,contLim=[-4540,3908],contour='sig_11')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress sig_22',warpScale=1,contLim=[-5075,9365],contour='sig_22')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress sig_33',warpScale=1,contLim=[25762,293000],contour='sig_33')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress sig_12',warpScale=1,contLim=[-2932,3116],contour='sig_12')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress sig_13',warpScale=1,contLim=[-35359,33715],contour='sig_13')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress sig_23',warpScale=1,contLim=[-57921,78221],contour='sig_23')
mlab.colorbar()

# Import EID Mesh CSV files to do element mapping
lam1AeroComBATEIDmesh = np.genfromtxt('lam1AeroComBATEIDmesh.csv', delimiter=',',dtype=int)
lam1NASTRANEIDmesh = np.genfromtxt('lam1NASTRANEIDmesh.csv', delimiter=',',dtype=int)

lam2AeroComBATEIDmesh = np.genfromtxt('lam2AeroComBATEIDmesh.csv', delimiter=',',dtype=int)
lam2NASTRANEIDmesh = np.genfromtxt('lam2NASTRANEIDmesh.csv', delimiter=',',dtype=int)

lam3AeroComBATEIDmesh = np.genfromtxt('lam3AeroComBATEIDmesh.csv', delimiter=',',dtype=int)
lam3NASTRANEIDmesh = np.genfromtxt('lam3NASTRANEIDmesh.csv', delimiter=',',dtype=int)

lam4AeroComBATEIDmesh = np.genfromtxt('lam4AeroComBATEIDmesh.csv', delimiter=',',dtype=int)
lam4NASTRANEIDmesh = np.genfromtxt('lam4NASTRANEIDmesh.csv', delimiter=',',dtype=int)

# Make mapping
NAST_2_Aero_EID = {}
for i in range(0,np.size(lam1AeroComBATEIDmesh,axis=0)):
    for j in range(0,np.size(lam1AeroComBATEIDmesh,axis=1)):
        NAST_2_Aero_EID[lam1NASTRANEIDmesh[i,j]]=lam1AeroComBATEIDmesh[i,j]

for i in range(0,np.size(lam2AeroComBATEIDmesh,axis=0)):
    for j in range(0,np.size(lam2AeroComBATEIDmesh,axis=1)):
        NAST_2_Aero_EID[lam2NASTRANEIDmesh[i,j]]=lam2AeroComBATEIDmesh[i,j]
        
for i in range(0,np.size(lam3AeroComBATEIDmesh,axis=0)):
    for j in range(0,np.size(lam3AeroComBATEIDmesh,axis=1)):
        NAST_2_Aero_EID[lam3NASTRANEIDmesh[i,j]]=lam3AeroComBATEIDmesh[i,j]

for i in range(0,np.size(lam4AeroComBATEIDmesh,axis=0)):
    for j in range(0,np.size(lam4AeroComBATEIDmesh,axis=1)):
        NAST_2_Aero_EID[lam4NASTRANEIDmesh[i,j]]=lam4AeroComBATEIDmesh[i,j]
# Import NASTRAN Stress Data:
NASTRANStress = np.genfromtxt('NASTRANStress.csv', delimiter=',')
# Determine L2 Norms
sigxx_L2 = max(abs(NASTRANStress[:,1]))
sigyy_L2 = max(abs(NASTRANStress[:,2]))
sigzz_L2 = max(abs(NASTRANStress[:,3]))
sigxy_L2 = max(abs(NASTRANStress[:,4]))
sigxz_L2 = max(abs(NASTRANStress[:,6]))
sigyz_L2 = max(abs(NASTRANStress[:,5]))

for i in range(0,np.size(NASTRANStress,axis=0)):
    AeroEID = NAST_2_Aero_EID[int(NASTRANStress[i,0])]
    tmpElem = xsect_Lay3.elemDict[AeroEID]
    sigxx = NASTRANStress[i,1]
    sigyy = NASTRANStress[i,2]
    sigzz = NASTRANStress[i,3]
    sigxy = NASTRANStress[i,4]
    sigxz = NASTRANStress[i,6]
    sigyz = NASTRANStress[i,5]
    tmpSigxx = np.mean(tmpElem.Sig[0,:])
    tmpSigyy = np.mean(tmpElem.Sig[1,:])
    tmpSigxy = np.mean(tmpElem.Sig[2,:])
    tmpSigxz = np.mean(tmpElem.Sig[3,:])
    tmpSigyz = np.mean(tmpElem.Sig[4,:])
    tmpSigzz = np.mean(tmpElem.Sig[5,:])
    SigxxError = (sigxx-tmpSigxx)/sigxx_L2*100
    SigyyError = (sigyy-tmpSigyy)/sigyy_L2*100
    SigzzError = (sigzz-tmpSigzz)/sigzz_L2*100
    SigxyError = (sigxy-tmpSigxy)/sigxy_L2*100
    SigxzError = (sigxz-tmpSigxz)/sigxz_L2*100
    SigyzError = (sigyz-tmpSigyz)/sigyz_L2*100
    
    tmpElem.Sig[0,:] = np.ones((1,4))*SigxxError
    tmpElem.Sig[1,:] = np.ones((1,4))*SigyyError
    tmpElem.Sig[2,:] = np.ones((1,4))*SigxyError
    tmpElem.Sig[3,:] = np.ones((1,4))*SigxzError
    tmpElem.Sig[4,:] = np.ones((1,4))*SigyzError
    tmpElem.Sig[5,:] = np.ones((1,4))*SigzzError

xsect_Lay3.plotWarped(figName='Stress Validation sig_11',warpScale=1,contLim=[-15,15],contour='sig_11')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress Validation sig_22',warpScale=1,contLim=[-15,15],contour='sig_22')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress Validation sig_33',warpScale=1,contLim=[-15,15],contour='sig_33')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress Validation sig_12',warpScale=1,contLim=[-15,15],contour='sig_12')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress Validation sig_13',warpScale=1,contLim=[-15,15],contour='sig_13')
mlab.colorbar()

xsect_Lay3.plotWarped(figName='Stress Validation sig_23',warpScale=1,contLim=[-15,15],contour='sig_23')
mlab.colorbar()

from AeroComBAT.AircraftParts import Wing
from AeroComBAT.FEM import Model

croot = 0.53
ctip = 0.53
x1 = -0.8990566037735849
x2 = 0.8990566037735849
n_ply = n_i_1+n_i_2+n_i_3+n_i_4
m_ply = m_i_1+m_i_2+m_i_3+m_i_4
th_ply = th_1+th_2+th_3+th_4
p1 = np.array([0.,0.,0.])
p2 = np.array([0.,0.,8.05])
Y_rib = np.linspace(0.,1.,2)
noe_dens = 31
chordVec=np.array([1.,0.,0.])
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_ply,matLib,name='box',\
    noe=noe_dens,chordVec=chordVec,ref_ax='origin',th_ply=th_ply,typeXSect='rectBox',n_orients=6)
sbeam1 = wing1.wingSects[0].SuperBeams[0]

# Make a FEM model
model  = Model()

model.addAircraftParts([wing1])

model.plotRigidModel(numXSects=10)

# Apply the constraint for the model
model.applyConstraints(0,'fix')

# CASE 1:
# Apply the case load
model.resetPointLoads()
tipLoad = np.array([100.,100.,10000.,0.,0.,0.])
F = {249:tipLoad}
model.applyLoads(1,F=F)
# Run the analysis
model.staticAnalysis(1)
model.plotDeformedModel(figName='V8 Case 1',numXSects=10,contLim=[0,293000],\
    warpScale=10,displScale=2,contour='sig_33')

# Composite Normal Mode Analysis
model.normalModesAnalysis()
freqs = model.freqs

model.plotDeformedModel(figName='Normal Mode 1',numXSects=10,contLim=[0,293000],\
    warpScale=25,displScale=10,contour='none',mode=1)
model.plotDeformedModel(figName='normalMode 2',numXSects=10,contLim=[0,293000],\
    warpScale=25,displScale=10,contour='none',mode=2)
model.plotDeformedModel(figName='normalMode 3',numXSects=10,contLim=[0,293000],\
    warpScale=25,displScale=10,contour='none',mode=3)
model.plotDeformedModel(figName='normalMode 4',numXSects=10,contLim=[0,293000],\
    warpScale=25,displScale=10,contour='none',mode=4)
model.plotDeformedModel(figName='normalMode 5',numXSects=10,contLim=[0,293000],\
    warpScale=25,displScale=10,contour='none',mode=5)

# Composite Deflections error
# Import NASTRAN Nodal Displacements
Layup3NastranNodalDisp = np.genfromtxt('Layup3NASTRANNodalDisplConstEnds.csv', delimiter=',')
# Create blank dictionary for displacements
T1 = {}
T2 = {}
T3 = {}
R1 = {}
R2 = {}
R3 = {}

for i in range(1,np.size(Layup3NastranNodalDisp,axis=0)):
    currentZ = Layup3NastranNodalDisp[i,3]
    if T1.has_key(currentZ):
        T1[currentZ] += [Layup3NastranNodalDisp[i,4]]
        T2[currentZ] += [Layup3NastranNodalDisp[i,5]]
        T3[currentZ] += [Layup3NastranNodalDisp[i,6]]
    else:
        T1[currentZ] = [Layup3NastranNodalDisp[i,4]]
        T2[currentZ] = [Layup3NastranNodalDisp[i,5]]
        T3[currentZ] = [Layup3NastranNodalDisp[i,6]]
# Calculate the average displacements
zposvec = T1.keys()
T1avg = []
T2avg = []
T3avg = []
for zpos, disp in T1.iteritems():
    T1avg += [np.mean(disp)]
for zpos, disp in T2.iteritems():
    T2avg += [np.mean(disp)]
for zpos, disp in T3.iteritems():
    T3avg += [np.mean(disp)]
    
zposvec,T1avg,T2avg,T3avg = zip(*sorted(zip(zposvec,T1avg,T2avg,T3avg)))

# Calculate the average rotations
for i in range(1,np.size(Layup3NastranNodalDisp,axis=0)):
    currentZ = Layup3NastranNodalDisp[i,3]
    zind = zposvec.index(currentZ)
    xold = Layup3NastranNodalDisp[i,1]
    yold = Layup3NastranNodalDisp[i,2]
    if R1.has_key(currentZ):
        if not abs(yold)<1e-6:
            R1[currentZ] += [np.arctan(Layup3NastranNodalDisp[i,6]/yold)]
        if not abs(xold)<1e-6:
            R2[currentZ] += [-np.arctan(Layup3NastranNodalDisp[i,6]/xold)]
        xnew = xold+Layup3NastranNodalDisp[i,4]-T1avg[zind]
        ynew = yold+Layup3NastranNodalDisp[i,5]-T2avg[zind]
        R3[currentZ] += [np.arccos((xnew*xold+ynew*yold)/(np.linalg.norm([xnew,ynew])\
            *np.linalg.norm([xold,yold])))]
    else:
        if not abs(yold)<1e-6:
            R1[currentZ] = [np.arctan(Layup3NastranNodalDisp[i,6]/yold)]
        if not abs(xold)<1e-6:
            R2[currentZ] = [-np.arctan(Layup3NastranNodalDisp[i,6]/xold)]
        xnew = xold+Layup3NastranNodalDisp[i,4]-T1avg[zind]
        ynew = yold+Layup3NastranNodalDisp[i,5]-T2avg[zind]
        R3[currentZ] = [np.arccos((xnew*xold+ynew*yold)/(np.linalg.norm([xnew,ynew])\
            *np.linalg.norm([xold,yold])))]

zposvec = R1.keys()
R1avg = []
R2avg = []
R3avg = []
for zpos, disp in R1.iteritems():
    R1avg += [np.mean(disp)]
for zpos, disp in R2.iteritems():
    R2avg += [np.mean(disp)]
for zpos, disp in R3.iteritems():
    R3avg += [np.mean(disp)]

zposvec,R1avg,R2avg,R3avg = zip(*sorted(zip(zposvec,R1avg,R2avg,R3avg)))

zposvec = np.array(zposvec)
T1avg = np.array(T1avg)
T2avg = np.array(T2avg)
T3avg = np.array(T3avg)
R1avg = np.array(R1avg)
R2avg = np.array(R2avg)
R3avg = np.array(R3avg)

AeroComBATData = sbeam1.writeDisplacements(Return=True)

ZposAero = AeroComBATData[:,3]
T1Aero = AeroComBATData[:,4]
T2Aero = AeroComBATData[:,5]
T3Aero = AeroComBATData[:,6]
R1Aero = AeroComBATData[:,7]
R2Aero = AeroComBATData[:,8]
R3Aero = AeroComBATData[:,9]

from matplotlib import pylab as pl

fig1 = pl.figure(1)
pl.hold(True)
pl.plot(ZposAero/8.05,T1Aero/8.05,'b-',label='AeroComBAT T1',linewidth=3)
pl.plot(ZposAero/8.05,T2Aero/8.05,'g-',label='AeroComBAT T2',linewidth=3)
pl.plot(ZposAero/8.05,T3Aero/8.05,'r-',label='AeroComBAT T3',linewidth=3)

pl.plot(zposvec/8.05,T1avg/8.05,'c--',label='NASTRAN T1',linewidth=3)
pl.plot(zposvec/8.05,T2avg/8.05,'m--',label='NASTRAN T2',linewidth=3)
pl.plot(zposvec/8.05,T3avg/8.05,'k--',label='NASTRAN T3',linewidth=3)
pl.grid(True)
pl.legend(loc=2,fontsize=10)
pl.title('Non-dimensional Beam Displacement')
pl.xlabel('Non-dimensional position along the beam')
pl.ylabel('Non-dimensional displacement')
pl.hold(False)

fig2 = pl.figure(2)
pl.hold(True)
pl.plot(ZposAero/8.05,R1Aero,'b-',label='AeroComBAT R1',linewidth=3)
pl.plot(ZposAero/8.05,R2Aero,'g-',label='AeroComBAT R2',linewidth=3)
pl.plot(ZposAero/8.05,R3Aero,'r-',label='AeroComBAT R3',linewidth=3)

pl.plot(zposvec/8.05,R1avg,'c--',label='NASTRAN R1',linewidth=3)
pl.plot(zposvec/8.05,R2avg,'m--',label='NASTRAN R2',linewidth=3)
pl.plot(zposvec/8.05,R3avg,'k--',label='NASTRAN R3',linewidth=3)
pl.grid(True)
pl.legend(loc=2,fontsize=10)
pl.title('Beam Rotation')
pl.xlabel('Non-dimensional position along the beam')
pl.ylabel('Rotation, rad')
pl.hold(False)
'''
def reorder(Q):
    scram_vec = [0,1,5,2,4,3]
    newMat = np.zeros((6,6))
    for i in range(0,np.size(Q,axis=0)):
        for j in range(0,np.size(Q,axis=1)):
            newMat[i,j] = Q[scram_vec[i],scram_vec[j]]
    return newMat

Q1 = xsect_Lay3.elemDict[0].Q
Q3 = xsect_Lay3.elemDict[1692].Q
Q2_p15 = xsect_Lay3.elemDict[1135].Q
Q2_m15 = xsect_Lay3.elemDict[1134].Q
Q4_p15 = xsect_Lay3.elemDict[2830].Q
Q4_m15 = xsect_Lay3.elemDict[2831].Q

Q1_Nast = reorder(Q1)
Q3_Nast = reorder(Q3)
Q2_p15_Nast = reorder(Q2_p15)
Q2_m15_Nast = reorder(Q2_m15)
Q4_p15_Nast = reorder(Q4_p15)
Q4_m15_Nast = reorder(Q4_m15)'''