# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:19:19 2016

@author: Ben
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:22:12 2016

@author: Ben
"""

from AeroComBAT.Structures import MaterialLib, Laminate, XSect
from AeroComBAT.Aerodynamics import Airfoil
import numpy as np

# ADD MATERIALS TO THE MATERIAL LIBRARY
# =====================================
# Create a material library object
matLib = MaterialLib()
matLib.addMat(1,'A-286','iso',[29.1e3,29.1e3/(11.1e3*2)-1,0.1],.005)



# NACA 2412 BOX BEAM
# ==================
# Now let's mesh a NACA2412 box beam. We will use the last of the supported
# meshing routines for this. This is the less restrictive than the 'rectBox'
# routine, and has different laminate mesh interfaces. This time we will also
# make a slightly more interesting mesh using unbalanced and unsymetric
# laminates. First let's initialize the airfoil shape:
# Initialize a chord length of four inches
# Initialize the non-dimesional locations for the airfoil points to be
# generated:
a = 0.91
b = 0.75
r = 0.437*2/3
xdim3 = [a,b,r]
n_i_1 = [1]
m_i_1 = [1]
lam1 = Laminate(n_i_1, m_i_1, matLib)
# Organize the laminates into an array
laminates_Lay3 = [lam1]
af3 = Airfoil(1.,name='NACA2412')
# Create the cross-section object and mesh it
xsect_Lay3 = XSect(4,af3,xdim3,laminates_Lay3,matLib,typeXSect='rectHole',nelem=40)
# Run the cross-sectional analysis. Since this is an airfoil and for this,
# symmetric airfoils the AC is at the 1/c chord, we will put the reference axis
# here
xsect_Lay3.xSectionAnalysis()#ref_ax=[0.25*c3,0.])
# Let's see what the rigid cross-section looks like:
xsect_Lay3.plotRigid()
# Print the stiffness matrix
xsect_Lay3.printSummary(stiffMat=True)

force3 = np.array([0.,0.,0.,0.,0.,216.])
# Calculate the force resultant effects
xsect_Lay3.calcWarpEffects(force=force3)
# This time let's plot the max principle stress:
xsect_Lay3.plotWarped(figName='sigma_xz',\
    warpScale=10,contour='sig_13',colorbar=True)
xsect_Lay3.plotWarped(figName='sigma_yz',\
    warpScale=10,contour='sig_23',colorbar=True)
xsect_Lay3.plotWarped(figName='RSS Shear',\
    warpScale=10,contour='rss_shear',colorbar=True)
xsect_Lay3.plotWarped(figName='Axial Stress',\
    warpScale=10,contour='sig_33',colorbar=True)