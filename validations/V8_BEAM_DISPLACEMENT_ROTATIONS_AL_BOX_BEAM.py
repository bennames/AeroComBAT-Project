# =============================================================================
# HEPHAESTUS VALIDATION 8 - BEAM DISPLACEMENTS AND ROTATIONS SIMPLE AL BOX BEAM
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib
from AeroComBAT.AircraftParts import Wing
import numpy as np
from AeroComBAT.FEM import Model

# Define the width of the cross-section
x1 = -0.8990566037735849
x2 = 0.8990566037735849
c = 1.
ctip = c
croot = c
p1 = np.array([0.,0.,0.])
p2 = np.array([0.,0.,20.])
Y_rib = np.linspace(0.,1.,2)
b_s = np.linalg.norm((Y_rib[0],Y_rib[-1]))

matLib = MaterialLib()
matLib.addMat(1,'AL','iso',[71.7e9,.33,2810],.005)
matLib.addMat(2,'Weak_mat','iso',[100,.33,10],.005)

n_ply = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
m_i = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

noe_dens = 2
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_i,matLib,name='box',noe=noe_dens)
sbeam1 = wing1.wingSects[0].SuperBeams[0]

model  = Model()

model.addAircraftParts([wing1])

model.plotRigidModel(numXSects=10)

# Apply the constraint for the model
model.applyConstraints(0,'fix')

# CASE 1:
# Apply the case load
tipLoad = np.array([-10000.,100000.,-300000.,35000.,60000.,10000.])
F = {40:tipLoad}
model.applyLoads(1,F=F)
# Run the analysis
model.staticAnalysis(1)
model.plotDeformedModel(figName='V8 Case 1',numXSects=10,contLim=[0,293000],\
    warpScale=10,displScale=2,contour='sig_33')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(fileName = 'V8_Case_1.csv')

# CASE 2:
# Apply the case load
def f(x):
    vx = -0.1*(-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    vy = (-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    pz = 0
    tz = .2*c*(-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    return np.array([vx,vy,pz,tz])/1.0e4
model.resetPointLoads()
wing1.applyLoads(f=f,allElems=True)
# Run the analysis
wing1.staticAnalysis(resetPointLoads=True)
wing1.plotDeformedWing(figName='V8 Case 2',numXSects=10,contLim=[0.,5.0e8],\
    warpScale=100,displScale=10,contour='MaxPrin')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(fileName = 'V8_Case_2.csv')

# CASE 3:
# Apply the case load
def f(x):
    vx = 1e3
    vy = 1e3
    pz = -1e3
    tz = 1e3
    return np.array([vx,vy,pz,tz])
wing1.applyLoads(f=f,allElems=True)
# Run the analysis
wing1.staticAnalysis(resetPointLoads=True)
wing1.plotDeformedWing(figName='V8 Case 3',numXSects=10,contLim=[0.,5.0e8],\
    warpScale=100,displScale=10,contour='MaxPrin')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(fileName = 'V8_Case_3.csv')

AeroComBAT_SOL = np.genfromtxt('V8_Case_3.csv', delimiter=',')
NASTRAN_SOL = np.genfromtxt('V8_Case_3_NASTRAN_SOL.csv', delimiter=',')

def u_analytical(z):
    return 5.74786e-6*z+0.0000144388*z**2-4.86082e-7*z**3+6.07603e-9*z**4
def v_analytical(z):
    return 0.0000136249*z+0.0000360233*z**2-1.21213e-6*z**3+1.51516e-8*z**4
def w_analytical(z):
    return -3.20696e-8*(40*z-z**2)
def psi_analytical(z):
    return -0.0000727279*z+3.6364e-6*z**2-6.06066e-8*z**3
def gamma_analytical(z):
    return 0.0000291649*z-1.45825e-6*z**2+2.43041e-8*z**3
def phi_analytical(z):
    return 2.18024e-7*(40*z-z**2)

z = np.linspace(0,20,41)

AeroComBAT_u = AeroComBAT_SOL[:,4]
AeroComBAT_v = AeroComBAT_SOL[:,5]
AeroComBAT_w = AeroComBAT_SOL[:,6]
AeroComBAT_psi = AeroComBAT_SOL[:,7]
AeroComBAT_gamma = AeroComBAT_SOL[:,8]
AeroComBAT_phi = AeroComBAT_SOL[:,9]

NASTRAN_u = NASTRAN_SOL[:,5]
NASTRAN_v = NASTRAN_SOL[:,6]
NASTRAN_w = NASTRAN_SOL[:,7]
NASTRAN_psi = NASTRAN_SOL[:,8]
NASTRAN_gamma = NASTRAN_SOL[:,9]
NASTRAN_phi = NASTRAN_SOL[:,10]

analytical_u = u_analytical(z)
analytical_v = v_analytical(z)
analytical_w = w_analytical(z)
analytical_psi = psi_analytical(z)
analytical_gamma = gamma_analytical(z)
analytical_phi = phi_analytical(z)

import matplotlib.pyplot as plt

plt.figure(1)
plt.hold(True)
plt.plot(z,(analytical_u-AeroComBAT_u)/analytical_u*100,'b-',label='T1 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_v-AeroComBAT_v)/analytical_v*100,'g-',label='T2 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_w-AeroComBAT_w)/analytical_w*100,'r-',label='T3 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_u-NASTRAN_u)/analytical_u*100,'c--',label='T1 NASTRAN Error',linewidth=3)
plt.plot(z,(analytical_v-NASTRAN_v)/analytical_v*100,'m--',label='T2 NASTRAN Error',linewidth=3)
plt.plot(z,(analytical_w-NASTRAN_w)/analytical_w*100,'k--',label='T3 NASTRAN Error',linewidth=3)
plt.legend()
plt.grid(True)
plt.title('Displacement Percent Error')
plt.xlabel('Position along the beam, m')
plt.ylabel('Percent error')
plt.hold(False)

plt.figure(2)
plt.hold(True)
plt.plot(z,(analytical_psi-AeroComBAT_psi)/analytical_psi*100,'b-',label='R1 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_gamma-AeroComBAT_gamma)/analytical_gamma*100,'g-',label='R2 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_phi-AeroComBAT_phi)/analytical_phi*100,'r-',label='R3 AeroComBAT Error',linewidth=3)
plt.plot(z,(analytical_psi-NASTRAN_psi)/analytical_psi*100,'c--',label='R1 NASTRAN Error',linewidth=3)
plt.plot(z,(analytical_gamma-NASTRAN_gamma)/analytical_gamma*100,'m--',label='R2 NASTRAN Error',linewidth=3)
plt.plot(z,(analytical_phi-NASTRAN_phi)/analytical_phi*100,'k--',label='R3 NASTRAN Error',linewidth=3)
plt.legend()
plt.grid(True)
plt.title('Rotation Percent Error')
plt.xlabel('Position along the beam, m')
plt.ylabel('Percent error')
axes = plt.gca()
axes.set_ylim([-.05,.15])
plt.hold(False)