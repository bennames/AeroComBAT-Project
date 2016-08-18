# =============================================================================
# AEROCOMBAT TUTORIAL 4 - Using Super Beams to Conduct Analysis
# =============================================================================

# IMPORT SYSTEM PACKAGES
# ======================
import sys
import os
sys.path.append(os.path.abspath('..'))

# IMPORT NUMPY PACKAGES
# =====================
import numpy as np
import matplotlib.pyplot as plt

# IMPORT AEROCOMBAT CLASSES
# =========================
from AeroComBAT.Structures import MaterialLib
from AeroComBAT.AircraftParts import Wing
from AeroComBAT.FEM import Model

# ADD MATERIALS TO THE MATERIAL LIBRARY
# =====================================
# Create a material library object
matLib = MaterialLib()
# Add an aluminum material (SI)
matLib.addMat(1,'AL','iso',[68.9e9,.33,2700*2],.00025)
# Add an soft material material (SI)
matLib.addMat(2,'Weak_mat','iso',[100,.33,10],.00025)
# Add material property from Hodges 1999 Asymptotically correct anisotropic
# beam theory (SI)
matLib.addMat(3,'AS43501-6*','trans_iso',[142e9,9.8e9,.34,.42,6e9,20000],0.0005)

# CREATE THE WING
# ===============
# Define the chord length of the model
c = .076
# Define the chord length (this will be the hight of the box beam)
ctip = 0.0076+.001
# Since the wing isn't tapered
croot = 0.0076+.001
# Define the non-dimensional starting point of the cross-section
x1 = -0.039/croot/2
# Define the non-dimensional ending point of the cross-section
x2 = 0.039/croot/2
# Define the span of the beam
span = 0.76
# Define the starting and stopping point of the wing structure
p1 = np.array([c/2,0.,0.])
p2 = np.array([c/2,span,0.])
# Define the non-dimesional locations of the ribs in the wing.
Y_rib = np.linspace(0.,1.,2)

# Initilize the layup schedule for the cross-section.
n_ply = [4,4,4,4]
# Initilize the material ID corresponding to an orientation in the
# cross-section
m_ply = [1,1,1,1]
# Initilize the orientations used in the box beam.
th_ply = [0,0,0,0]

# Define the number of orientations used in each laminate. In this case, we'll
# just use one.
n_orients = 1
# Define the number of laminates per cross-section. In this case since we are
# using a box beam, it will be 4.
n_lams = 4
# Define the type of cross-section. In this case it'll be the 'rectBox' mesh
typeXSect = 'rectBox'
# Define the number of elements per unit length are to be used. The structure
# will have then 120*span beam elements.
noe_dens = 120
# Define the aditional vector required to define the orientation of the beam.
# In this case, we'll have it pointing down the length of the wing.
chordVec=np.array([1.,0.,0.])
# Create the wing object. For more information about some of the input
# parameters see the AeroComBAT documentation
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_ply,matLib,name='box',\
    noe=noe_dens,chordVec=chordVec,ref_ax='shearCntr',n_orients=n_orients,\
    n_lams=n_lams,typeXSect=typeXSect,meshSize=2,th_ply=th_ply)
# This is an optional step for ease of programming. Since only one wing section
# was created and the wing isn't tapered, there is only one superbeam which
# contains all of the beam elements in the wing.
sbeam1 = wing1.wingSects[0].SuperBeams[0]

# ADD A LIFTING SURFCE TO THE WING
# ================================
# Define the root leading edge location
x1 = np.array([0,0.,0.])
# Define the root trailing edge location
x2 = np.array([c,0.,0.])
# Define the tip trailing edge location
x3 = np.array([c,span,0.])
# Define the tip leading edge location
x4 = np.array([0,span,0.])
# Determine the number of boxes to be used in the spanwise direction
nspan = 36*2
# Determine the number of boxes to be used in the chordwise direction
nchord = 10
# Add the lifting surface to the wing.
wing1.addLiftingSurface(1,x1,x2,x3,x4,nspan,nchord)

# MAKE THE FINITE ELEMENT MODEL (FEM)
# ===================================
model  = Model()
# Add the aircraft wing to the model
model.addAircraftParts([wing1])
# Apply the constraint for the model
model.applyConstraints(0,'fix')
# Plot the rigid wing in the finite elment model
model.plotRigidModel(numXSects=10)
# Conduct a normal modes analysis. Since the normal mode shapes are used in the
# flutter analysis it is good to run this ahead of time to make sure you are
        # selecting enough mode shapes to include any relevant torsional and bending
# mode shapes.
model.normalModesAnalysis()
# Save the frequencies of the modal analysis
freqs = model.freqs

# IMPORT NASTRAN RESULTS
# ======================
# The same analysis was conducted on a plate model in NX NASTRAN to verify the
# results produced by AeroComBAT
NASTRAN = np.genfromtxt('NASTRANFlutterResults.csv', delimiter=',')
UNAST = NASTRAN[:,0]
Damp1 = NASTRAN[:,1]
Freq1 = NASTRAN[:,2]
Damp2 = NASTRAN[:,3]
Freq2 = NASTRAN[:,4]
Damp3 = NASTRAN[:,5]
Freq3 = NASTRAN[:,6]
Damp4 = NASTRAN[:,7]
Freq4 = NASTRAN[:,8]
Damp5 = NASTRAN[:,9]
Freq5 = NASTRAN[:,10]
Damp6 = NASTRAN[:,11]
Freq6 = NASTRAN[:,12]

# CONDUCT FLUTTER ANALYSIS
# ========================
# Whenever a flutter analysis is conducted, several quantities need to be
# defined. The first is an array of free-stream airspeeds:
U_vec = np.linspace(1,342,100)
# In addition, a vector of trial reduced frequencies must be initialized. Keep
# in mind that the because the non-looping pk method is used, a wide range of
# reduced frequencies may need to be used.
kr_vec = np.array([0.,1e-06,1e-04,.001,.01,.05,.1,.5,1.,5.,10.,50])*10
# A vector of mach numbers must also be used. These should be kept close to
# The suspected flutter mach number. If mach numbers greater than 0.7 are used,
# is is likely doublet lattice method is no-longer valid.
M_vec = [0.]*len(kr_vec)
# Initialize the sea level density
rho_0 = 1.225
# Determine the number of modes to be used
nmodes = 6
# Run the flutter analysis. Depending on how many reduced frequencies and
# velocities sampled, this could take anywhere from a 30 seconds to 2 minutes.
model.flutterAnalysis(U_vec,kr_vec,M_vec,c,rho_0,nmodes,symxz=True,g=0.0)

# POST-PROCESS THE FLUTTER RESULTS
# ================================
# Note that in these figures, the dashed lines are the NASTRAN results, whereas
# the solid lines are the AeroComBAT results.
cvec = ['b','g','r','c','m','y']
plt.figure(1)
plt.hold(True)
for PID, point in model.flutterPoints.iteritems():
    plt.plot(U_vec,point.gamma,color=cvec[PID],label='Mode '+str(PID+1))
plt.legend(loc=2)
plt.ylim([-1,1])
plt.plot(UNAST,Damp1,str(cvec[0])+'--',label='NASTRAN Mode 1')
plt.plot(UNAST,Damp2,str(cvec[1])+'--',label='NASTRAN Mode 2')
plt.plot(UNAST,Damp3,str(cvec[2])+'--',label='NASTRAN Mode 3')
plt.plot(UNAST,Damp4,str(cvec[3])+'--',label='NASTRAN Mode 4')
plt.plot(UNAST,Damp5,str(cvec[4])+'--',label='NASTRAN Mode 5')
plt.plot(UNAST,Damp6,str(cvec[5])+'--',label='NASTRAN Mode 6')
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
plt.plot(UNAST,Freq1,str(cvec[0])+'--',label='NASTRAN Mode 1')
plt.plot(UNAST,Freq2,str(cvec[1])+'--',label='NASTRAN Mode 2')
plt.plot(UNAST,Freq3,str(cvec[2])+'--',label='NASTRAN Mode 3')
plt.plot(UNAST,Freq4,str(cvec[3])+'--',label='NASTRAN Mode 4')
plt.plot(UNAST,Freq5,str(cvec[4])+'--',label='NASTRAN Mode 5')
plt.plot(UNAST,Freq6,str(cvec[5])+'--',label='NASTRAN Mode 6')
plt.title('Frequency of the Wing Modes')
plt.xlabel('Free-stream airspeed, m/s')
plt.ylabel('Mode Frequency, Hz')
plt.grid(True)
plt.hold(False)

# The following figures demonstrate how the damping and frequencies are
# interpolated.
Uind = 80
point1 = model.flutterPoints[0]
point2 = model.flutterPoints[1]
point3 = model.flutterPoints[2]
point4 = model.flutterPoints[3]
omegaAeros = point1.omegaAeroDict[U_vec[Uind]]
omegaRoots1 = point1.omegaRootDict[U_vec[Uind]]
omegaRoots2 = point2.omegaRootDict[U_vec[Uind]]
omegaRoots3 = point3.omegaRootDict[U_vec[Uind]]
omegaRoots4 = point4.omegaRootDict[U_vec[Uind]]
gammas1 = point1.gammaDict[U_vec[Uind]]
gammas2 = point2.gammaDict[U_vec[Uind]]
gammas3 = point3.gammaDict[U_vec[Uind]]
gammas4 = point4.gammaDict[U_vec[Uind]]

plt.figure(3)
plt.hold(True)
plt.plot(omegaAeros,omegaAeros,str(cvec[0])+'o-',label='omega_aero')
plt.plot(omegaAeros,omegaRoots1,str(cvec[1])+'o-',label='omega_root_1')
plt.plot(omegaAeros,omegaRoots2,str(cvec[2])+'o-',label='omega_root_2')
plt.plot(omegaAeros,omegaRoots3,str(cvec[3])+'o-',label='omega_root_3')
plt.plot(omegaAeros,omegaRoots4,str(cvec[4])+'o-',label='omega_root_4')
plt.legend(loc=2)
plt.ylim([0,20000])
plt.xlim([0,25000])
plt.xlabel('Aerodynamic frequency, rad')
plt.ylabel('Root frequency, rad')
plt.title('Interpolation of Root Requencies at V=%4.2f m/s'%(U_vec[Uind]))
plt.grid(True)
plt.hold(False)

plt.figure(4)
plt.hold(True)
plt.plot(omegaAeros,gammas1,str(cvec[1])+'o-',label='gamma_root_1')
plt.plot(omegaAeros,gammas2,str(cvec[2])+'o-',label='gamma_root_2')
plt.plot(omegaAeros,gammas3,str(cvec[3])+'o-',label='gamma_root_3')
plt.plot(omegaAeros,gammas4,str(cvec[4])+'o-',label='gamma_root_4')
plt.legend(loc=3)
plt.ylim([-1,.1])
plt.xlim([0,25000])
plt.xlabel('Aerodynamic frequency, rad')
plt.ylabel('Damping (g)')
plt.title('Interpolation of Root Damping at V=%4.2f m/s'%(U_vec[Uind]))
plt.grid(True)
plt.hold(False)