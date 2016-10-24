# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:22:12 2016

@author: Ben
"""

from AeroComBAT.Structures import MaterialLib, Laminate, XSect, SuperBeam
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np
from AeroComBAT.FEM import Model

# ADD MATERIALS TO THE MATERIAL LIBRARY
# =====================================
# Create a material library object
matLib = MaterialLib()
matLib.addMat(1,'IM7 Uni Tape','trans_iso',[22950000.,1360000.,.34,.3,620000.,0.058],.005)
matLib.addMat(2,'31IG Rohacell','trans_iso',[5220.,5220.,.34,.3,1885.,.1],.19/2)
# Add an aluminum material (SI)
matLib.addMat(3,'A-286','iso',[29.1e3,29.1e3/(11.1e3*2)-1,0.1],.005)



# NACA 2412 BOX BEAM
# ==================
# Now let's mesh a NACA2412 box beam. We will use the last of the supported
# meshing routines for this. This is the less restrictive than the 'rectBox'
# routine, and has different laminate mesh interfaces. This time we will also
# make a slightly more interesting mesh using unbalanced and unsymetric
# laminates. First let's initialize the airfoil shape:
# Initialize a chord length of four inches
c3 = 17.68515
# Initialize the non-dimesional locations for the airfoil points to be
# generated:
xdim3 = [.19,.582]
# Create the airfoil object:
af3 = Airfoil(c3,name='NACA2412')
# Create the laminates to make up the cross-section
n_i_1 = [8]
m_i_1 = [1]
th_1 = [10]
lam1 = Laminate(n_i_1, m_i_1, matLib, th=th_1)
n_i_2 = [1,1,1,1]
m_i_2 = [1,1,1,1]
th_2 = [45,-45,45,-45]
lam2 = Laminate(n_i_2, m_i_2, matLib, th=th_2)
n_i_3 = [8]
m_i_3 = [1]
th_3 = [-10]
lam3 = Laminate(n_i_3, m_i_3, matLib, th=th_3)
n_i_4 = [1,1,1,1]
m_i_4 = [1,1,1,1]
th_4 = [45,-45,45,-45]
lam4 = Laminate(n_i_4, m_i_4, matLib, th=th_4)
# Organize the laminates into an array
laminates_Lay3 = [lam1,lam2,lam3,lam4]
# Create the cross-section object and mesh it
xsect_Lay3 = XSect(4,af3,xdim3,laminates_Lay3,matLib,typeXSect='box',meshSize=4)
# Run the cross-sectional analysis. Since this is an airfoil and for this,
# symmetric airfoils the AC is at the 1/c chord, we will put the reference axis
# here
xsect_Lay3.xSectionAnalysis()#ref_ax=[0.25*c3,0.])
K_tmp = xsect_Lay3.K
F_tmp = xsect_Lay3.F
xs_tmp = xsect_Lay3.xs
ys_tmp = xsect_Lay3.ys
# Let's see what the rigid cross-section looks like:
xsect_Lay3.plotRigid()
# Print the stiffness matrix
xsect_Lay3.printSummary(stiffMat=True)

x1 = np.array([0,0,0])
x2 = np.array([0,0,160.5337])
# Initialize a superbeam ID
SBID = 1
# Next we need to initialize the number of elements the superbeam should mesh:
noe = 40
# Now let's make the superbeam
sbeam1 = SuperBeam(SBID,x1,x2,xsect_Lay3,noe)

model = Model()
# Easy right? Now let's add the superbeam to the model.
model.addElements([sbeam1])
# Now that our beam is loaded into the FEM, let's visualize it!
model.plotRigidModel(numXSects=8)
# First let's constrain the beam at it's root. Since we only added one
# superbeam and we never specified a starting node ID, we know that the first
# node ID is actually 1! So when we constrain the model, we can just select:
model.applyConstraints(1,'fix')
model.normalModesAnalysis(analysis_name='normal modes')
# For now, the frequencies can be accessed via the model attribute 'freqs'
Frequencies = model.freqs
# Let's plot the first mode:
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 1',numXSects=10,contour='none',\
    warpScale=1,displScale=50,mode=1)
# How about the second?
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 2',numXSects=10,contour='none',\
    warpScale=1,displScale=50,mode=2)
# How about the third? It happens to be the torsional mode.
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 3',numXSects=10,contour='none',\
    warpScale=1,displScale=25,mode=3)
model.plotDeformedModel(analysis_name='normal modes',\
    figName='Normal Modes 3',numXSects=10,contour='none',\
    warpScale=1,displScale=25,mode=3)