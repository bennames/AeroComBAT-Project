# =============================================================================
# AEROCOMBAT TUTORIAL 4 - Using Super Beams to Conduct Analysis
# =============================================================================

# IMPORT SYSTEM PACKAGES
# ======================
import sys
import os

sys.path.append(os.path.abspath('..'))

# IMPORT AEROCOMBAT CLASSES
# =========================
from AeroComBAT.Structures import MaterialLib, Laminate, XSect, SuperBeam
from AeroComBAT.Aerodynamics import Airfoil
from AeroComBAT.FEM import Model

# IMPORT NUMPY MODULES
# ====================
import numpy as np
import mayavi.mlab as mlab

# ADD MATERIALS TO THE MATERIAL LIBRARY
# =====================================
# Create a material library object
matLib = MaterialLib()
# Add material property from Hodges 1999 Asymptotically correct anisotropic
# beam theory (Imperial)
matLib.addMat(1,'AS43501-6','trans_iso',[20.6e6,1.42e6,.34,.3,.87e6,.1],.005)
# Add material property from Hodges 1999 Asymptotically correct anisotropic
# beam theory (Imperial)
matLib.addMat(2,'AS43501-6*','trans_iso',[20.6e6,1.42e6,.34,.42,.87e6,.1],.005)
# Add an aluminum material (SI)
matLib.addMat(3,'AL','iso',[71.7e9,.33,2810],.005)

# CREATE A WRAPPED RECTANGULAR BOX-BEAM CROSS-SECTION
# ===================================================
# Layup 1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [0]_6)

# Before we make a beam, we must first make the cross-section of that beam. We
# are going to start with a cross-section we used in the third tutorial.
c2 = 0.53
# Establish the non-dimensional starting and stopping points of the cross-
# section.
xdim2 = [-.953/(c2*2),.953/(c2*2)]
# Generate the airfoil box:
af2 = Airfoil(c2,name='box')
# Now let's make all of the laminate objects we will need for the box beam. In
# this case it's 4:
n_i_Lay1 = [6]
m_i_Lay1 = [2]
th_Lay1 = [0.]
lam1_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam2_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam3_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
lam4_Lay1 = Laminate(n_i_Lay1, m_i_Lay1, matLib, th=th_Lay1)
# Assemble the laminates into an array.
laminates_Lay1 = [lam1_Lay1,lam2_Lay1,lam3_Lay1,lam4_Lay1]
# Create the cross-section vector:
xsect_Lay1 = XSect(2,af2,xdim2,laminates_Lay1,matLib,typeXSect='rectBox',\
    meshSize=2)
# Create the cross-section object.
xsect_Lay1.xSectionAnalysis()
# Having created the cross-section, we can now generate a superbeam. A
# superbeam is just a collection of beam elements. In other words, a superbeam
# is just there to fascilitate beam meshing and other pre/post processing
# benefits. In order to make a superbeam, we need to initialize a few things.
# First, let's initialize the starting and stopping location of the beam:
x1 = np.array([0,0,0])
x2 = np.array([0,0,4])
# Initialize a superbeam ID
SBID = 1
# Next we need to initialize the number of elements the superbeam should mesh:
noe = 20
# Now let's make the superbeam
sbeam1 = SuperBeam(SBID,x1,x2,xsect_Lay1,noe)
# In order to analyze this beam, we'll need to add it to a finite element
# model. First let's make a finite element model!
model = Model()
# Easy right? Now let's add the superbeam to the model.
model.addElements([sbeam1])
# Now that our beam is loaded into the FEM, let's visualize it!
model.plotRigidModel(numXSects=8)
# First let's constrain the beam at it's root. Since we only added one
# superbeam and we never specified a starting node ID, we know that the first
# node ID is actually 1! So when we constrain the model, we can just select:
model.applyConstraints(1,'fix')
# There are two supported keywords for constraints, either 'fix' or 'pinned'.
# If the user wanted to apply a different constraint, they can just enter the
# degrees of freedom to constrain on the model. This can be done by supplying
# an array such as: [1,2,3,5]. Now let's apply a load. We will make two load
# cases. In the first case, we are going to apply a simple tip load:
load1 = {21:np.array([100.,100.,0.,0.,0.,100.])}
# We can also create a function for a distributed load:
def load2(x):
   vx = (1/10)*10*x[2]**2-7*x[2]-2.1
   vy = 10*x[2]**2-7*x[2]
   pz = 0
   mx = 0
   my = 0
   tz = (10*x[2]**2-7*x[2])/10+3*x[0]**2
   return np.array([vx,vy,pz,mx,my,tz])
# Ok now let's add these loads to the model:
model.applyLoads(1,F=load1)
# Notice that when I applied a tip load, I did it using the argument 'F'. When
# we apply a distributed load function, we use the argument 'f' instead.
model.applyLoads(2,f=load2,allElems=True)
# Now with constraints and loads, we can run a static analysis! Let's run the
# first load case.
model.staticAnalysis(1,analysis_name='tip load')
# Let's see what results we get:
model.plotDeformedModel(analysis_name='tip load',figName='Tip Load Analysis',\
    numXSects=8,contour='VonMis',contLim=[0,1e5],warpScale=50,displScale=10)
# Now let's try analyzing the distributed load:
model.staticAnalysis(2,analysis_name='distributed load')
# Let's see what results we get for the distributed load:
model.plotDeformedModel(analysis_name='distributed load',\
    figName='Distributed Load Analysis',numXSects=8,contour='VonMis',\
    contLim=[0,1e5],warpScale=50,displScale=10)
# Really quickly, let's discuss some of the keywords used in the analysis and
# plotting. The analysis_name designates the name where the results should be
# stored for the beam. Therefore if you want to keep multiple results stored
# at once you can give it a name. Otherwise this will always result to the
# default name. figName is just the name of the MayaVi figure, and numXSects
# designates how many cross-sections should be plotted. Note that the more
# cross-sections are plotted, the slower the plotting process will be. Both
# contour, contLim and warpScale were discussed in Tutorial 3. displScale is
# the scalling factor applied to the beam displacements and rotations.
# We can also run a normal modes analysis on the beam as well: having already
# applied the constraints, we can run:
model.normalModesAnalysis(analysis_name='normal modes')
# For now, the frequencies can be accessed via the model attribute 'freqs'
Frequencies = model.freqs
# Let's plot the first mode:
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 1',numXSects=10,contour='none',\
    warpScale=1,displScale=2,mode=1)
# How about the second?
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 2',numXSects=10,contour='none',\
    warpScale=1,displScale=5,mode=2)
# How about the third? It happens to be the torsional mode.
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 3',numXSects=10,contour='none',\
    warpScale=1,displScale=1,mode=3)
