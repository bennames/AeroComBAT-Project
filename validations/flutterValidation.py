# =============================================================================
# HEPHAESTUS VALIDATION 4 - MESHER AND CROSS-SECTIONAL ANALYSIS
# =============================================================================

# IMPORTS:

import sys
import os
import cProfile


sys.path.append(os.path.abspath('..'))

from AeroComBAT.Structures import MaterialLib
from AeroComBAT.AircraftParts import Wing
from AeroComBAT.FEM import Model
import numpy as np

# Add the material property
matLib = MaterialLib()
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.3,.34,.87e6,0.057/386.09],0.005)
matLib.addMat(3,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,0.057/386.09],0.005)
matLib.addMat(2,'AL','iso',[9900000.,.33,2.53881E-4],.005)

# Layup 3 Configuration
n_i_1 = [1,1,1,1,1,1]
m_i_1 = [2,2,2,2,2,2]
th_1 = [-15,-15,-15,-15,-15,-15]
n_i_2 = [1,1,1,1,1,1]
m_i_2 = [2,2,2,2,2,2]
th_2 = [15,-15,15,-15,15,-15]
n_i_3 = [1,1,1,1,1,1]
m_i_3 = [2,2,2,2,2,2]
th_3 = [15,15,15,15,15,15]
n_i_4 = [1,1,1,1,1,1]
m_i_4 = [2,2,2,2,2,2]
th_4 = [-15,15,-15,15,-15,15]
# Concatenated layup schedule arrays
n_ply = n_i_1+n_i_2+n_i_3+n_i_4
m_ply = m_i_1+m_i_2+m_i_3+m_i_4
th_ply = th_1+th_2+th_3+th_4

# Define the chord length of the box beam
croot = 0.53
ctip = 0.53
# Define nd starting and stopping points of the cross-section
x1 = -0.8990566037735849
x2 = 0.8990566037735849
p1 = np.array([0.,0.,0.])
p2 = np.array([0.,8.05,0.])
Y_rib = np.linspace(0.,1.,2)
noe_dens = 6
chordVec=np.array([-1.,0.,0.])
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_ply,matLib,name='box',\
    noe=noe_dens,chordVec=chordVec,ref_ax='origin',th_ply=th_ply,typeXSect='rectBox',n_orients=6)
sbeam1 = wing1.wingSects[0].SuperBeams[0]

# Add lifting surface to wing
x1 = np.array([-4*croot,0.,0.])
x2 = np.array([4*croot,0.,0.])
x3 = np.array([4*croot,p2[1],0.])
x4 = np.array([-4*croot,p2[1],0.])
nspan = 36/2
nchord = 20/2

wing1.addLiftingSurface(1,x1,x2,x3,x4,nspan,nchord)

# Make a FEM model
model  = Model()

model.addAircraftParts([wing1])

model.plotRigidModel(numXSects=10)

# Apply the constraint for the model
model.applyConstraints(0,'fix')
#cProfile.run('model.normalModesAnalysis()',sort='tottime')

# Composite Normal Mode Analysis
model.normalModesAnalysis()
freqs = model.freqs
'''
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
'''

# Flutter Analysis
# Note units are inches, seconds, and pounds
U_vec = np.linspace(1,100,100)
kr_vec = np.array([.001,.005,.01,.05,.1,.5,1.,5.,10.,50.,100.,500.,1000.,5000.])
M_vec = [0.]*len(kr_vec)
# In slinch
rho_0 = 1.225
nmodes = 6

model.jitflutterAnalysis(U_vec,kr_vec,M_vec,croot*4,rho_0,nmodes,symxz=True,g=.01)

# flutter plots
import matplotlib.pyplot as plt
plt.figure(1)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.gamma,label='mode'+str(PID))
plt.legend(loc=3)
#plt.ylim([-.001,150])
plt.grid(True)
plt.hold(False)

plt.figure(2)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.omega,label='mode'+str(PID))
plt.legend(loc=1)
#plt.ylim([0,150])
plt.grid(True)
plt.hold(False)