"""
Created on Tue Feb 23 09:03:21 2016

@author: Ben
"""
import sys
import os
sys.path.append(os.path.abspath('..\..'))

# =============================================================================
# HEPHAESTUS VALIDATION 8 - BEAM DISPLACEMENTS AND ROTATIONS SIMPLE AL BOX BEAM
# =============================================================================

# IMPORTS:
import numpy as np
from AeroComBAT.Structures import MaterialLib
from AeroComBAT.AircraftParts import Wing
from AeroComBAT.FEM import Model

# Define the width of the cross-section
c = .076
ctip = 0.0076+.001
croot = 0.0076+.001
x1 = -0.039/croot/2
x2 = 0.039/croot/2
span = 0.76#.305*3/2
p1 = np.array([c/2,0.,0.])
p2 = np.array([c/2,span,0.])
Y_rib = np.linspace(0.,1.,2)
b_s = np.linalg.norm((Y_rib[0],Y_rib[-1]))

matLib = MaterialLib()
matLib.addMat(1,'AL','iso',[68.9e9,.33,2700*2],.00025)
matLib.addMat(2,'Weak_mat','iso',[100,.33,10],.00025)
matLib.addMat(3,'AS43501-6*','trans_iso',[142e9,9.8e9,.34,.42,6e9,1580],.00025)

n_ply = [4,4,4,4]
m_ply = [3,3,3,3]
th_ply = [-22.9,0,22.9,0]
#n_ply = [4]
#m_ply = [1]

n_orients = 1
n_lams = 4
typeXSect = 'rectBox'
noe_dens = 120
chordVec=np.array([1.,0.,0.])


# For tension bending coupling
#m_ply = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
#th_ply = [0,0,0,0,-30,-30,-30,-30,0,0,0,0,30,30,30,30]


wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_ply,matLib,name='box',\
    noe=noe_dens,chordVec=chordVec,ref_ax='origin',n_orients=n_orients,\
    n_lams=n_lams,typeXSect=typeXSect,meshSize=1,th_ply=th_ply)
sbeam1 = wing1.wingSects[0].SuperBeams[0]


x1 = np.array([0,0.,0.])
x2 = np.array([c,0.,0.])
x3 = np.array([c,span,0.])
x4 = np.array([0,span,0.])
nspan = 36*2
nchord = 10

wing1.addLiftingSurface(1,x1,x2,x3,x4,nspan,nchord)

# Make a FEM model
model  = Model()

model.addAircraftParts([wing1])

# Apply the constraint for the model
model.applyConstraints(0,'fix')
model.plotRigidModel(numXSects=10)

model.normalModesAnalysis()
freqs = model.freqs
'''
model.plotDeformedModel(mode=1,figName='mode 1',numXSects=10,displScale=2)
model.plotDeformedModel(mode=2,figName='mode 2',numXSects=10,displScale=2)
model.plotDeformedModel(mode=3,figName='mode 3',numXSects=10,displScale=2)
model.plotDeformedModel(mode=4,figName='mode 4',numXSects=10,displScale=2)
model.plotDeformedModel(mode=5,figName='mode 5',numXSects=10,displScale=2)
model.plotDeformedModel(mode=6,figName='mode 6',numXSects=10,displScale=2)
model.plotDeformedModel(mode=7,figName='mode 7',numXSects=10,displScale=2)
'''
'''

model.plotDeformedModel(mode=1,figName='mode 1',numXSects=10,displScale=2)
model.plotDeformedModel(mode=2,figName='mode 2',numXSects=10,displScale=2)
model.plotDeformedModel(mode=3,figName='mode 3',numXSects=10,displScale=2)
model.plotDeformedModel(mode=4,figName='mode 4',numXSects=10,displScale=2)
model.plotDeformedModel(mode=5,figName='mode 5',numXSects=10,displScale=2)

# Aerodynamic Model Validation
model.calcAIC(.24,.47,.4572/2,symxz=True)

model.normalModesAnalysis()

pbar = np.dot(np.linalg.inv(model.D),np.dot(model.W,-model.umode[:,0]/max(abs(model.umode[:,0]))))
pbarReal = np.real(pbar)
pbarImag = np.imag(pbar)

pbarPlot = pbar[330:330+10]
#pbarPlot = pbar[0:0+10]

xs = np.zeros((10))
for i in range(0,10):
    xs[i] = model.aeroBox[i].Xr[0]
xs = xs/c

import matplotlib.pyplot as plt
plt.figure(1)
plt.hold(True)
plt.plot(xs,np.real(pbarPlot),label='real part')
plt.plot(xs,np.imag(pbarPlot),label='imaginary part')
plt.legend()
plt.grid(True)
plt.hold(False)
'''
'''
# CASE 1:
# Apply the case load
model.resetPointLoads()
tipLoad = np.array([10000.,0.,0.,0.,0.,0.])
F = {80:tipLoad}
model.applyLoads(1,F=F)
# Run the analysis
model.staticAnalysis(1)
model.plotDeformedModel(figName='V8 Case 1',numXSects=8,contLim=[-4.0e6,4.0e6],\
    warpScale=100,displScale=10,contour='sig_33')
'''

U_vec = np.linspace(1,300,250)
kr_vec = np.array([0.,1e-05,.001,.01,.05,.1,.5,1.,5.,10.,50])*10
M_vec = [0.]*len(kr_vec)
rho_0 = 1.225
nmodes = 7

model.flutterAnalysis(U_vec,kr_vec,M_vec,c,rho_0,nmodes,symxz=True,g=0.0)

# flutter plots
import matplotlib.pyplot as plt
cvec = ['b','g','r','c','m','y','k']
plt.figure(1)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.gamma,color=cvec[PID],label='Mode '+str(PID+1))
plt.legend(loc=2)
plt.ylim([-1,1])
plt.title('Damping of the Wing Modes')
plt.xlabel('Free-stream airspeed, m/s')
plt.ylabel('Damping, g')
plt.grid(True)
plt.hold(False)

plt.figure(2)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.omega,color = cvec[PID],label='Mode '+str(PID+1))
plt.legend(loc=1)
#plt.ylim([0,150])
plt.title('Frequency of the Wing Modes')
plt.xlabel('Free-stream airspeed, m/s')
plt.ylabel('Mode Frequency, Hz')
plt.grid(True)
plt.hold(False)


flutterResults1 = np.genfromtxt('CUSparametricFlutter.csv', delimiter=',',dtype=float)
th1 = flutterResults1[:,0]
flutterSpeed1 = flutterResults1[:,1]
plt.figure(3)
plt.plot(th1,flutterSpeed1)
plt.title('Parametric Flutter Speed')
plt.xlabel('Orientation of fibers, degrees')
plt.ylabel('Flutter speed, m/s')
plt.grid(True)

flutterResults2 = np.genfromtxt('CASparametricFlutter.csv', delimiter=',',dtype=float)
th2 = flutterResults2[:,0]
flutterResults2 = flutterResults2[:,1]
plt.figure(4)
plt.plot(th2,flutterResults2)
plt.title('Parametric Flutter Speed')
plt.xlabel('Orientation of fibers, degrees')
plt.ylabel('Flutter speed, m/s')
plt.xlim([-90,90])
plt.xticks(np.arange(min(th2), max(th2)+1, 15.0))
plt.grid(True)
