# =============================================================================
# HEPHAESTUS VALIDATION 8 - BEAM DISPLACEMENTS AND ROTATIONS SIMPLE AL BOX BEAM
# =============================================================================

# IMPORTS:
import numpy as np
from Structures import MaterialLib
from AircraftParts import Wing
from FEM import Model

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
matLib.addMat(3,'AS43501-6*','trans_iso',[142e9,9.8e9,.34,.42,6e9,2000],0.005)

n_ply = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
m_ply = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# For tension bending coupling
#m_ply = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
#th_ply = [0,0,0,0,-30,-30,-30,-30,0,0,0,0,30,30,30,30]

noe_dens = 4
chordVec=np.array([1.,0.,0.])
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_ply,matLib,name='box',\
    noe=noe_dens,chordVec=chordVec,ref_ax='origin')#,th_ply=th_ply)
sbeam1 = wing1.wingSects[0].SuperBeams[0]


x1 = np.array([-0.8990566037735849*2,0.,0.])
x2 = np.array([0.8990566037735849*2,0.,0.])
x3 = np.array([0.8990566037735849*2,20.,0.])
x4 = np.array([-0.8990566037735849*2,20.,0.])
nspan = 8*5
nchord = 2*5

#wing1.addLiftingSurface(1,x1,x2,x3,x4,nspan,nchord)

# Make a FEM model
model  = Model()

model.addAircraftParts([wing1])

# Apply the constraint for the model
model.applyConstraints(0,'fix')

model.plotRigidModel(numXSects=10)


#model.calcAIC(.24,.47,.4572/2,symxz=0)

# CASE 1:
# Apply the case load
model.resetPointLoads()
tipLoad = np.array([0.,10000.,0.,0.,0.,0.])
F = {80:tipLoad}
model.applyLoads(1,F=F)
# Run the analysis
model.staticAnalysis(1)
model.plotDeformedModel(figName='V8 Case 1',numXSects=8,contLim=[-4.0e6,4.0e6],\
    warpScale=10000,displScale=1000,contour='sig_33')

'''
U_vec = np.linspace(5,100,20)
kr_vec = np.array([.001,.01,.1,.5,2.5,25.])*10
M_vec = [0.,0.,0.,0.,0.,0.,0.]
rho_0 = 1.225
nmodes = 8

model.flutterAnalysis(U_vec,kr_vec,M_vec,0.8990566037735849*2,rho_0,nmodes,symxz=True)

# flutter plots
import matplotlib.pyplot as plt
plt.figure(1)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.gamma,label='mode'+str(PID))
plt.legend()
plt.grid(True)
plt.hold(False)

plt.figure(2)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.omega,label='mode'+str(PID))
plt.legend()
#plt.ylim([8,15])
plt.grid(True)
plt.hold(False)

model.normalModesAnalysis()
model.plotDeformedModel(mode=1,figName='mode 1',numXSects=10,displScale=5)
model.plotDeformedModel(mode=2,figName='mode 2',numXSects=10,displScale=5)
model.plotDeformedModel(mode=3,figName='mode 3',numXSects=10,displScale=5)
model.plotDeformedModel(mode=4,figName='mode 4',numXSects=10,displScale=5)
model.plotDeformedModel(mode=5,figName='mode 5',numXSects=10,displScale=5)
'''
'''
import cProfile
# Initialize an array of PANIDs
PANIDs = model.aeroBox.keys()
# Initialize the number of panels
numPan = len(PANIDs)
Area = np.zeros((numPan,numPan))
# For all the recieving panels
for i in range(0,numPan):
    recievingBox = model.aeroBox[PANIDs[i]]
    Area[i,i] = recievingBox.Area
model.AeroArea = Area
cProfile.run('model.calcAIC(0.,1.,0.8990566037735849*2)')
'''
'''
# Test Normal modes
model.normalModesAnalysis(analysis_name='Normal_Modes')
# Write the beam displacements and rotations to a file
freqs = model.freqs
model.plotDeformedModel(mode=1,figName='Mode 1',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
model.plotDeformedModel(mode=2,figName='Modes 2',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
model.plotDeformedModel(mode=3,figName='Modes 3',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
model.plotDeformedModel(mode=4,figName='Modes 4',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
model.plotDeformedModel(mode=5,figName='Modes 5',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
model.plotDeformedModel(mode=6,figName='Modes 6',analysis_name='Normal_Modes',\
    numXSects=10,warpScale=1,displScale=2)
'''
'''
# CASE 2:
# Apply the case load
def f(x):
    vx = -0.1*(-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    vy = (-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    pz = 0
    tz = .2*c*(-1.0e3*x[2]**2+6e7*x[2]+1.0e6)
    return np.array([vx,vy,pz,tz])/1.0e5
wing1.applyLoads(f=f,allElems=True)
# Run the analysis
wing1.staticAnalysis(resetPointLoads=True)
wing1.plotDeformedWing(figName='V8 Case 2',numXSects=10,contLim=[0.,5.0e8],\
    warpScale=10,displScale=10,contour='MaxPrin')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(fileName = 'V8_Case_2.csv')

# CASE 3:
# Apply the case load
def f(x):
    vx = 1e3
    vy = 1e3
    pz = -1e3
    tz = 1e3
    return np.array([vx,vy,pz,tz])/1e1
wing1.applyLoads(f=f,allElems=True)
# Run the analysis
wing1.staticAnalysis(resetPointLoads=True)
wing1.plotDeformedWing(figName='V8 Case 3',numXSects=10,contLim=[0.,5.0e8],\
    warpScale=100,displScale=10,contour='MaxPrin')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(fileName = 'V8_Case_3.csv')'''