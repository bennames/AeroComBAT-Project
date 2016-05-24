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
import scipy as sci
from scipy.sparse.linalg import gmres
from memory_profiler import profile

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

# Establish Matricies
A = xsect_Lay3.A
R = xsect_Lay3.R
E = xsect_Lay3.E
C = xsect_Lay3.C
L = xsect_Lay3.L
M = xsect_Lay3.M
D = xsect_Lay3.D
Z6 = np.zeros((6,6))
nd = 3*len(xsect_Lay3.nodeDict)
Tr = np.zeros((6,6));Tr[0,4] = -1;Tr[1,3] = 1

@profile
def luDecomp():
    EquiA1 = np.vstack((np.hstack((E,R,D)),np.hstack((R.T,A,Z6)),\
                                    np.hstack((D.T,Z6,Z6))))                                    
    # Assemble solution vector for first equation
    Equib1 = np.vstack((np.zeros((nd,6)),Tr.T,Z6))
    # LU factorize state matrix as it will be used twice
    lu,piv = sci.linalg.lu_factor(EquiA1,check_finite=False)
    EquiA1 = 0
    # Solve system
    sol1 = sci.linalg.lu_solve((lu,piv),Equib1,check_finite=False,overwrite_b=True)
    # Recover gradient of displacement as a function of force and moment
    # resutlants
    dXdz = sol1[0:nd,:]
    # Save the gradient of section strains as a function of force and
    # moment resultants
    dYdz = sol1[nd:nd+6,:]
    # Set up the first of two solution vectors for second equation
    Equib2_1 = np.vstack((np.hstack((-(C-C.T),L)),np.hstack((-L.T,Z6)),np.zeros((6,nd+6))))
    # Set up the second of two solution vectors for second equation
    Equib2_2 = np.vstack((np.zeros((nd,6)),np.eye(6,6),Z6))
    # Add solution vectors and solve second equillibrium equation
    sol2 = sci.linalg.lu_solve((lu,piv),np.dot(Equib2_1,sol1[0:nd+6,:])+Equib2_2,check_finite=False,overwrite_b=True)
    X = sol2[0:nd,0:6]
    # Store the section strain as a function of force and moment resultants
    Y = sol2[nd:nd+6,0:6]
    return dXdz, dYdz, X, Y
    
@profile
def xsectAnalysis():
    xsect_Lay3.xSectionAnalysis(ref_ax=[0.25*c3,0.])
    
#test1, test2, test3, test4 = luDecomp()
#xsectAnalysis()

def GMRES():
    EquiA1 = np.vstack((np.hstack((E,R,D)),np.hstack((R.T,A,Z6)),\
                                    np.hstack((D.T,Z6,Z6))))                                    
    # Assemble solution vector for first equation
    Equib1 = np.vstack((np.zeros((nd,6)),Tr.T,Z6))
    sol1 = np.zeros((nd+12,6))
    for i in range(6):
        sol1[:,i] = gmres(EquiA1,Equib1[:,i])
    sol1 = gmres(EquiA1,Equib1)
    # Recover gradient of displacement as a function of force and moment
    # resutlants
    dXdz = sol1[0:nd,:]
    # Save the gradient of section strains as a function of force and
    # moment resultants
    dYdz = sol1[nd:nd+6,:]
    # Set up the first of two solution vectors for second equation
    Equib2_1 = np.vstack((np.hstack((-(C-C.T),L)),np.hstack((-L.T,Z6)),np.zeros((6,nd+6))))
    # Set up the second of two solution vectors for second equation
    Equib2_2 = np.vstack((np.zeros((nd,6)),np.eye(6,6),Z6))
    # Add solution vectors and solve second equillibrium equation
    sol2 = np.zeros((nd+12,6))
    Equib2 = np.dot(Equib2_1,sol1[0:nd+6,:])+Equib2_2
    for i in range(6):
        sol2[:,i] = gmres(EquiA1,Equib2[:,i])
    X = sol2[0:nd,0:6]
    # Store the section strain as a function of force and moment resultants
    Y = sol2[nd:nd+6,0:6]
    return dXdz, dYdz, X, Y