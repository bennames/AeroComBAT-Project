# =============================================================================
# HEPHAESTUS VALIDATION 8 - BEAM DISPLACEMENTS AND ROTATIONS SIMPLE AL BOX BEAM
# =============================================================================

# IMPORTS:
import numpy as np
from Structures import MaterialLib
from AircraftParts import Wing
import matplotlib.pyplot as pl

# Define the width of the cross-section
x1 = -0.8990566037735849
x2 = 0.8990566037735849
c = 1.
ctip = c
croot = c
L = 20.
Y_rib = np.linspace(0.,L,2)
b_s = np.linalg.norm((Y_rib[0],Y_rib[-1]))

# Initialize material Info
matLib = MaterialLib()
matLib.addMat(1,'AL','iso',[71.7e9,.33,2810],.005)

n_ply = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
m_i = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

noe_dens = [.1,.2,.4,.6,.8,1.,1.2,1.4,1.6,2.,3.,4.,5.]
wings = []
sbeams = []
displArr = []

# Apply the case load
def f(x):
    vx = 1e3
    vy = 1e3
    pz = -1e3
    tz = 1e3
    return np.array([vx,vy,pz,tz])
    

for noe in noe_dens:
    # Create Wing
    tmpWing = Wing(b_s,croot,ctip,x1,x2,Y_rib,n_ply,m_i,matLib,\
        name='box',noe_per_unit_length=noe)
    wings+=[tmpWing]
    # Save Super Beam
    tmpSBeam = tmpWing.wingSects[0].SuperBeams[0]
    sbeams+=[tmpSBeam]
    # Fix End of Wing
    tmpWing.addConstraint(0,'fix')
    # Load Wing
    tmpWing.applyLoads(f=f,eid=range(0,int(L*noe)))
    # Run Analysis
    tmpWing.staticAnalysis(resetPointLoads=True)
    # Recover displacements and rotations:
    displArr+=[tmpSBeam.writeDisplacements(Return=True)]

# Establish analytical answers:
def u(z):
    return 5.74786e-6*z + 0.0000144388*z**2 - 4.86082e-7*z**3 + 6.07603e-9*z**4
def v(z):
    return 0.0000136249*z + 0.0000360233*z**2 - 1.21213e-6*z**3 + 1.51516e-8*z**4
def w(z):
    return -3.20696e-8*(40*z - z**2)
def psi(z):
    return -0.0000727279*z + 3.6364e-6*z**2 - 6.06066e-8*z**3
def gam(z):
    return -(-0.0000291649*z + 1.45825e-6*z**2 - 2.43041e-8*z**3)
def phi(z):
    return 2.18024e-7*(40*z - z**2)

fig = pl.figure(num=1,figsize=(15,15))
test = []
c = ['b','g','r','c','m','y','k']
dof = ['o', 'v', '^', '<', '>', '*']
pl.hold(True)
j = 0
offset = 2
for i in range(offset,len(c)+offset):
    displ = displArr[i]
    num_elem = np.size(displ,axis=0)-1
    z_node = displ[:,3]
    T1 = displ[:,4]
    T2 = displ[:,5]
    T3 = displ[:,6]
    R1 = displ[:,7]
    R2 = displ[:,8]
    R3 = displ[:,9]
    pl.plot(z_node,np.log((u(z_node)-T1)/u(z_node)*100),label='T1 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[0])
    pl.plot(z_node,np.log((v(z_node)-T2)/v(z_node)*100),label='T2 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[1])
    pl.plot(z_node,np.log((w(z_node)-T3)/w(z_node)*100),label='T3 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[2])
    pl.plot(z_node,np.log((psi(z_node)-R1)/psi(z_node)*100),label='R1 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[3])
    pl.plot(z_node,np.log((gam(z_node)-R2)/gam(z_node)*100),label='R2 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[4])
    pl.plot(z_node,np.log((phi(z_node)-R3)/phi(z_node)*100),label='R3 Error: '+str(num_elem)+' elements',color=c[j],marker=dof[5])
    test = displ
    j+=1
pl.legend(loc=2,fontsize=8,ncol=len(range(offset,len(c)+offset)))
#pl.ylim((-1,2))
#pl.xlim((-1,21))
pl.hold(False)
pl.show()


tmpBeam = wings[9].wingSects[0].SuperBeams[0]
tmpBeam.writeDisplacements(fileName = 'tmp.csv')
'''
wing1 = Wing(b_s,croot,ctip,x1,x2,Y_rib,n_ply,m_i,matLib,name='box',noe_per_unit_length=noe_dens)
sbeam1 = wing1.wingSects[0].SuperBeams[0]

# Apply the constraint for the model
wing1.addConstraint(0,'fix')


wing1.applyLoads(f=f,eid=range(0,int(L*noe_dens)))
# Run the analysis
wing1.staticAnalysis(resetPointLoads=True)
wing1.plotDeformedWing(figName='V8 Case 7',numXSects=4,contLim=[0.,5.0e8],\
    warpScale=100,displScale=10,contour='MaxPrin')
# Write the beam displacements and rotations to a file
sbeam1.writeDisplacements(Return=True)'''