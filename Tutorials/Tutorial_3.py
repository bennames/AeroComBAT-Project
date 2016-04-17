# =============================================================================
# AEROCOMBAT TUTORIAL 3 - Using XSect Objects
# =============================================================================

# IMPORT SYSTEM PACKAGES
# ======================
import sys
import os

sys.path.append(os.path.abspath('..'))

# IMPORT AEROCOMBAT CLASSES
# =========================
from AeroComBAT.Structures import MaterialLib, Laminate, XSect
from AeroComBAT.Aerodynamics import Airfoil

# IMPORT NUMPY MODULES
# ====================
import numpy as np


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

# CREATE A LAMINATE CROSS-SECTION
# ===============================
# Create a box airfoil object. Note that when creating an airfoil object, only
# the chord length is used. As such, it doesn't truly matter if the airfoil
# has an airfoil profile or a box profile. In this case we will just give it a
# box profile.
# Initialize the chord length
c1 = 1.
# Initialize the non-dimensional starting and stopping points of the cross-
# section. These bounds when dimensionalized will determine the overall
# dimesions of the cross-section. Therefore the total width of the laminate is:
# xdim[1]*c1-xdim[0]*x. In this case, the total width is 2!
xdim1 = [-1.,1.]
af1 = Airfoil(c1,name='box')
# Create a layup schedule for the laminate. In this case, we will select a
# layup schedule of [0_2/45/90/3]_s
th_1 = [0,45,90]
n_1 = [2,1,3]
m_1 = [1,1,1]
# Notice how the orientations are stored in the 'th_1' array, the subscripts are
# stored in the 'n_1' array, and the material information is held in 'm_1'.
# Create the laminate object:
lam1 = Laminate(n_1,m_1,matLib,th=th_1,sym=True)
# In order to make a cross-section, we must add all of the laminates to be used
# to an array:
laminates1 = [lam1]
# We now have all the information necessary to make a laminate beam cross-
# section:
xsect1 = XSect(1,af1,xdim1,laminates1,matLib,typeXSect='laminate',meshSize=2)
# With the cross-section object initialized, let's run the cross-sectional
# analysis to get cross-section stiffnesses, etc.
xsect1.xSectionAnalysis()
# Let's see what our rigid cross-section looks like when plotted in 3D:
xsect1.plotRigid(mesh=True)
# Note that while it might look like the cross-section is made of triangular
# elements, it's actually made of quadrilaterals. This is an artifact of how
# the visualizer mayavi works. Let's get a summary of the cross-section's
# stiffnesses, ect.
xsect1.printSummary(stiffMat=True)
# Notice that from the command line output, all of the important cross-
# sectional geometric properties are located at the origin. By observing the
# cross-section stiffness matrix, it can be seen by the 1,3 entry that there
# is shear-axial coupling. From the non-zero 4,6 entry, we can also tell that
# the cross-section has bending-torsion coupling.
# We can apply a force to the face of this cross-section at the reference axis
# (which in this case is at x,y = 0,0) and see what the stresses look like. In
# this case we'll apply [Fx,Fy,Fz,Mx,My,Mz]=[0.,0.,0.,100.,0.,0.] as if the
# beam belonging to this cross-section were in pure bending.
force1 = np.array([0.,100.,0.,10.,0.,0.])
xsect1.calcWarpEffects(force=force1)
# Having applied the force, let's see what the sigma_zz (normal stresses of the
# beam) look like
xsect1.plotWarped(figName='Laminate Sigma_33 Stress',warpScale=10,\
    contour='sig_33',colorbar=True)
# Let's look at the sigma_13 stress state now since we know there is torsion
# coupling:
xsect1.plotWarped(figName='Laminate Sigma_13 Stress',warpScale=10,\
    contour='sig_13',colorbar=True)
# Notice the increased stress in two of the plies? Recall which ones those are?
# Those plies are the 45 degree plies which are currently taking the shear!

# CREATE A LAMIANTE CROSS-SECTION WITH A DIFFERENT REFERENCE AXIS
# ===============================================================
# The cross-section we just made happened to have all of it's geometrical
# locations (mass center, shear center, tension center) at the origin, which
# is where we applied our force resultant. Suppose we wanted to give the cross-
# section a different reference axis. We can do this by executing the
# xSectionAnalysis method again:
ref_ax = [.5,0.]
# This will move the reference axis (location where we apply forces and
# moments) to x=0.5, y=0. Carrying out the cross-sectional analysis again, we
# get:
xsect1.xSectionAnalysis(ref_ax=ref_ax)
# Let's see how the cross-section's stiffness matrix has changed:
xsect1.printSummary(stiffMat=True)

# Let's not apply the same force resultant and see what the stresses look like:
xsect1.calcWarpEffects(force=force1)
xsect1.plotWarped(figName='Laminate Sigma_33 Stress New Reference Axis',\
    warpScale=10,contour='sig_33',colorbar=True)
xsect1.plotWarped(figName='Laminate Sigma_13 Stress New Reference Axis',\
    warpScale=10,contour='sig_13',colorbar=True)
# Notice how the stress resultants are fairly different once we moved the
# reference axis.

# CREATE A WRAPPED RECTANGULAR BOX-BEAM CROSS-SECTION
# ===================================================
# Layup 1 Box beam (0.5 x 0.923 in^2 box with laminate schedule [0]_6)

# Let's make a slightly more complex cross-section. Using the 'rectBox' key
# word and four laminates, we can make a cross-section box-beam. In the next
# case, we are going to make a cross-section from Hodges 1999 Asymptotically
# correct anisotropic beam theory paper. We will do the most simple box-beam
# in this case, which is the "Layup 1" case:
# Establish the chord length
c2 = 0.53
# Establish the non-dimensional starting and stopping points of the cross-
# section.
xdim2 = [-.953/(c2*2),.953/(c2*2)]
# This can be confirmed by plotting, but it should be noted that for the
# 'rectBox' routine, the mesh y-coordinates will go from -c2/2 -> c2/2, and the
# x-coordinates will go from xdim2[0]*c -> xdim2[1]*c. Therefore the box's
# overall dimesions should be from 0.53 in x 0.953 in. Next we will generate
# the airfoil box:
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
# Assemble the laminates into an array. Refer back to the documentation to
# remind yourself of stacking sequence direction, etc. It should be noted that
# the first laminate is at the top of the box cross-section. The next is the
# left laminate, and so on in a counter-clockwise direction.
laminates_Lay1 = [lam1_Lay1,lam2_Lay1,lam3_Lay1,lam4_Lay1]
# Create the cross-section vector:
xsect_Lay1 = XSect(2,af2,xdim2,laminates_Lay1,matLib,typeXSect='rectBox',\
    meshSize=2)
# Create the cross-section object. Note that since we aren't specifying the
# reference axis, it will automatically be set at the shear center. Since this
# cross-section is simple, this will still be at the origin.
xsect_Lay1.xSectionAnalysis()
# Let's look at the stiffness matrix:
xsect_Lay1.printSummary(stiffMat=True)
# Since this cross-section is simple, we can analytically calculate some of the
# simpler parameters. For example, the 3,3 entry is just E1*A. Similarly, the
# bending stiffnesses of the cross-section are just E1*I_xx and E1*I_yy. Try
# calculating them on your own to verify this! Note that this will only work
# since all of the fibers have a 0 degree orientation. Let's apply a load and
# see how it behaves!
# Force Resultant Vector:
force2 = np.array([0.,0.,0.,0.,0.,100.])
# Calculate the effects of the force resultant
xsect_Lay1.calcWarpEffects(force=force2)
# Plot the normal stress
xsect_Lay1.plotWarped(figName='Layup 1 Box Beam Sigma_33 Stress',\
    warpScale=100,contour='sig_33',colorbar=True)
# Now the shear 13 stress!
xsect_Lay1.plotWarped(figName='Layup 1 Box Beam Sigma_13 Stress',\
    warpScale=100,contour='sig_13',colorbar=True)

# Look at the differences in magnitudes of the stress between the two plots.
# Notice anything? There is virtually no normal stress, but A LOT of shear
# stress. This makes sense though since we only applied a torque to the cross-
# section. Note that this warping profile is vero common for any box type cross
# section. Let's try one more slightly more complex shape:

# NACA 2412 BOX BEAM
# ==================

# Now let's mesh a NACA2412 box beam. We will use the last of the supported
# meshing routines for this. This is the less restrictive than the 'rectBox'
# routine, and has different laminate mesh interfaces. This time we will also
# make a slightly more interesting mesh using unbalanced and unsymetric
# laminates. First let's initialize the airfoil shape:
# Initialize a chord length of four inches
c3 = 4.
# Initialize the non-dimesional locations for the airfoil points to be
# generated:
xdim3 = [.15,.7]
# Create the airfoil object:
af3 = Airfoil(c3,name='NACA2412')
# Create the laminates to make up the cross-section
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
# Organize the laminates into an array
laminates_Lay3 = [lam1,lam2,lam3,lam4]
# Create the cross-section object and mesh it
xsect_Lay3 = XSect(4,af3,xdim3,laminates_Lay3,matLib,typeXSect='box',meshSize=2)
# Run the cross-sectional analysis. Since this is an airfoil and for this,
# symmetric airfoils the AC is at the 1/c chord, we will put the reference axis
# here
xsect_Lay3.xSectionAnalysis(ref_ax=[0.25*c3,0.])
# Let's see what the rigid cross-section looks like:
xsect_Lay3.plotRigid()
# Print the stiffness matrix
xsect_Lay3.printSummary(stiffMat=True)
# Create an applied force vector. For a wing shape such as this, let's apply a
# semi-realistic set of loads:
force3 = np.array([10.,100.,0.,10.,1.,0.])
# Calculate the force resultant effects
xsect_Lay3.calcWarpEffects(force=force3)
# This time let's plot the max principle stress:
xsect_Lay3.plotWarped(figName='NACA2412 Box Beam Max Principle Stress',\
    warpScale=10,contour='MaxPrin',colorbar=True)
