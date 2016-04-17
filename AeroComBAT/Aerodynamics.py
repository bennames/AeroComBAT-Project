# Aerodynamics.py
# Author: Ben Names
"""
This module contains a library of classes devoted to modeling aircraft parts.

The main purpose of this library is to model various types of aircraft parts.
Currently only wing objects are suported, however in the future it is possible
that fuselages as well as other parts will be added.

:SUMARRY OF THE METHODS:

- `K`: The kernel function used in the doublet-lattice method to relate
    downwashes to panel pressures.
- `calcAIC`: Provided several vectors of numbers as well as a reduced frequency
    and mach number, this method calculates a matrix of AIC's using doublet-
    lattice method elementary solutions. This method is used by the FEM class
    flutterAnalysis method.
    
:SUMARRY OF THE CLASSES:

- `Airfoil`: Primarily used for the generation of structural cross-sectional
    meshes, this class represent an airfoil. This class could be expanded in
    future to use simple 2D panel methods for an airfoil of arbitrary shape.
- `CQUADA`: This class creates quadrilateral panels intended to be used for
    potential flow panel methods. Currently it is used for the unsteady
    doublet-lattice panels.
- `CAERO1`: This class is used to generate a lattice of CQUADA panels.

"""
__docformat__ = 'restructuredtext'
# =============================================================================
# IMPORT ANACONDA ASSOCIATED MODULES
# =============================================================================
import numpy as np
import mayavi.mlab as mlab
from numba import jit
# =============================================================================
# IMPORT ADDITIONAL MODULES
# =============================================================================
from tabulate import tabulate
# =============================================================================
# DEFINE AeroComBAT AERODYNAMIC CLASSES
# =============================================================================

# Define Constants for later sumation integration
an = np.array([.24186198,-2.7918027,24.991079,-111.59196,271.43549,-305.75288,\
    -41.18363,545.98537,-644.78155,328.72755,-64.279511])
c = 0.372
n = np.array(range(1,12))
# Define Kernel Function Here
@jit(nopython=True)
def K(Xr,Xs,gamma_r,gamma_s,M,br,kr,r1):
    """Evaluates the doublet-lattice kernel function.
    
    Provided several geometric parameters about the sending and recieving
    panels, this method evaluates the kernel function which relates the
    pressure on one panel to the downwash induced at another panel.
    
    :Args:
    
    - `Xr (1x3 np.array[float])`: The location of the recieving point.
    - `Xs (1x3 np.array[float])`: The location of the sending point.
    - `gamma_r (1x3 np.array[float])`: The dihedral of the panel corresponding
        to the recieving point.
    - `gamma_s (1x3 np.array[float])`: The dihedral of the panel corresponding
        to the sending point.
    - `M (float)`: The mach number
    - `br (float)`: The reference semi-chord
    - `kr (float)`: The reduced frequency
    - `r1 (float)`: The scalar distance between the sending point and the
        recieving point.
        
    :Returns:
    
    - `Kbar (complex128)`: The evaluation of the unsteady kernel function which
        is complex in nature.
    """
    # Vector pointing from sending node to recieving node
    x0 = Xr[0]-Xs[0]
    y0 = Xr[1]-Xs[1]
    z0 = Xr[2]-Xs[2]
    
    # Check if r1 is very small
    if abs(r1)<br/10000.:
        if x0>0:
            return 2*np.exp(-1j*x0*kr/br)
        else:
            return 0.
    # Prandtl-Glauert Compressability Correction Factor
    beta = (1-M**2)**(0.5)
    # Reduced Frequency
    k1 = r1*kr/br
    # Another distance value?
    R = np.sqrt(x0**2+beta**2*r1**2)
    # If r1 is not very small:
    u1 = (M*R-x0)/(beta**2*r1)
    T1 = np.cos(gamma_r-gamma_s)
    T2 = (z0*np.cos(gamma_r)-y0*np.sin(gamma_r))\
        *(z0*np.cos(gamma_s)-y0*np.sin(gamma_s))/r1**2
    if abs(T1)<1e-6:
        K1=0.
    else:
        if u1>=0:
            I1_val = I1(u1,k1)
        else:
            I1_val = 2*I1(0,k1).real\
                -I1(-u1,k1).real+1j*I1(-u1,k1).imag
        K1 = I1_val+M*r1*np.exp(-1j*k1*u1)/(R*np.sqrt(1+u1**2))
    if abs(T2)<1e-6:
        K2 = 0.
    else:
        if u1>=0:
            I2_3_val = I2_3(u1,k1)
        else:
            I2_3_val = 2*I2_3(0,k1).real\
                -I2_3(-u1,k1).real+1j*I2_3(-u1,k1).imag
        K2 = -I2_3_val\
            -1j*k1*M**2*r1**2/R**2*np.exp(-1j*k1*u1)/(1+u1**2)**(0.5)\
            -M*r1/R*((1+u1**2)*beta**2*r1**2/R**2+2+M*r1*u1/R)\
            *np.exp(-1j*k1*u1)/(1+u1**2)**(1.5)
    return np.exp(-1j*kr*x0/br)*(K1*T1+K2*T2)#-T1*(1.+x0/R)-T2*(-2-x0/R*(2+beta**2*r1**2/R**2))
# Definition of I0 integral
@jit(nopython=True)
def I0(u1,k1):
    I_0 = 0.
    #I_0 = np.dot(an,np.exp(-n*c*u1)/(n**2*c**2+k1**2)*(n*c-1j*k1))
    for i in range(len(n)):
        I_0+=an[i]*np.exp(-n[i]*c*u1)/(n[i]**2*c**2+k1**2)*(n[i]*c-1j*k1)
    return I_0
# Definition of I1 integral
@jit(nopython=True)
def I1(u1,k1):
    return np.exp(-1j*k1*u1)*(1-u1/np.sqrt(1+u1**2)-1j*k1*I0(u1,k1))
# Definition of J0 integral
@jit(nopython=True)
def J0(u1,k1):
    J_0 = 0.
    #J_0 = np.dot(an,np.exp(-n*c*u1)/(n**2*c**2+k1**2)**2\
    #        *(n**2*c**2-k1**2+n*c*u1*(n**2*c**2+k1**2)\
    #        -1j*k1*(2*n*c+u1*(n**2*c**2+k1**2))))
    for i in range(0,len(n)):
        J_0+=an[i]*np.exp(-n[i]*c*u1)/(n[i]**2*c**2+k1**2)**2\
            *(n[i]**2*c**2-k1**2+n[i]*c*u1*(n[i]**2*c**2+k1**2)\
            -1j*k1*(2*n[i]*c+u1*(n[i]**2*c**2+k1**2)))
    return J_0
# Definition of 3*I2 integral
@jit(nopython=True)
def I2_3(u1,k1):
    return np.exp(-1j*k1*u1)*((2+1j*k1*u1)*(1-u1/(1+u1**2)**(0.5))\
        -u1/(1+u1**2)**(1.5)-1j*k1*I0(u1,k1)+k1**2*J0(u1,k1))

# Functions for JIT calcAIC Method
@jit(nopython=True)
def eta(yr,ys,zr,zs,gamma_s):
    return (yr-ys)*np.cos(gamma_s)+(zr-zs)*np.sin(gamma_s)

@jit(nopython=True)
def zeta(yr,ys,zr,zs,gamma_s):
    return -(yr-ys)*np.sin(gamma_s)+(zr-zs)*np.cos(gamma_s)
    
@jit(nopython=True)
def I_plan(A,B,C,e,eta_0):
    return (eta_0**2*A+eta_0*B+C)*(1./(eta_0-e)-1./(eta_0+e))+\
                    (B/2+eta_0*A)*np.log(((eta_0-e)/(eta_0+e))**2)
                    
@jit(nopython=True)
def I_nonplan(A,B,C,e,eta_0,zeta_0,r1):
    return ((eta_0**2-zeta_0**2)*A+eta_0*B+C)*zeta_0**(-1)*\
                    np.arctan(2*e*abs(zeta_0)/(r1**2-e**2))+\
                    (B/2+eta_0*A)*np.log((r1**2-2*eta_0*e+e**2)/\
                    (r1**2+2*eta_0*e+e**2))+2*e*A

@jit(nopython=True)
def calcAIC(M,kr,br,delta_x_vec,sweep_vec,l_vec,dihedral_vec,Xr_vec,Xi_vec,Xc_vec,\
    Xo_vec,symxz=False):
    """Calculate the doublet-lattice AIC's.
    
    Provided the geometry of all of the doublet-lattice panels, this method
    calculates the AIC matrix.
    
    :Args:
    
    - `M (float)`: The mach number.
    - `kr (float)`: The reduced frequency.
    - `br (float)`: The reference semi-chord.
    - `delta_x_vec (1xN array[float]`: An array of chord length of the panels.
    - `sweep_vec (1xN array[float])`: An array of sweep angles of the panels.
    - `l_vec (1xN array[float])`: An array of average doublet line lengths of
        the panels.
    - `dihedral_vec (1xN array[float])`: An array of dihedral angles of the
        panels.
    - `Xr_vec (Nx3 np.array[float])`: A matrix of recieving points, where a row
        are the 3D coordinates of the point.
    - `Xi_vec (Nx3 np.array[float])`: A matrix of inboard sending points, where
        a row are the 3D coordinates of the point.
    - `Xc_vec (Nx3 np.array[float])`: A matrix of center sending points, where
        a row are the 3D coordinates of the point.
    - `Xo_vec (Nx3 np.array[float])`: A matrix of outboard sending points,
        where a row are the 3D coordinates of the point.
    - `symxz (bool)`: A boolean operater intended to determine whether or not
        a reflection of the panels should be considered over the xz-plane.
        
    :Returns:
    
    - `D (NPANxNPAN np.array[complex128])`: The matrix which relates pressures
        over panels to induced velocities over those panels. In more simple
        terms, this is the inverse of the desired AIC matrix.
    """
    # Initialize the number of panels
    numPan = len(Xr_vec)
    # Initialize the complex [D] matrix
    D = np.zeros((numPan,numPan),dtype=np.complex128)
    # For all the recieving panels
    for i in range(0,numPan):
        # For all the sending panels
        Xr = Xr_vec[i,:]
        gamma_r = dihedral_vec[i]
        for j in range(0,numPan):
            
            #sendingBox = self.aeroBox[PANIDs[j]]
            # Calculate average chord of sending box
            delta_x_j = delta_x_vec[j]
            # Calculate sweep of sending box
            lambda_j = sweep_vec[j]
            # Calculate the length of the doublet line on sending box
            l_j = l_vec[j]
            Xi = Xi_vec[j,:]
            Xc = Xc_vec[j,:]
            Xo = Xo_vec[j,:]
            e = 0.5*l_j*np.cos(lambda_j)
            gamma_s = dihedral_vec[j]
            eta_0 = eta(Xr[1],Xc[1],Xr[2],Xc[2],gamma_s)
            zeta_0 = zeta(Xr[1],Xc[1],Xr[2],Xc[2],gamma_s)
            r1 = np.sqrt(eta_0**2+zeta_0**2)
            # Calculate the Kernel function at the inboard, middle, and
            # outboard locations
            Ki = K(Xr,Xi,gamma_r,gamma_s,M,br,kr,r1)
            Kc = K(Xr,Xc,gamma_r,gamma_s,M,br,kr,r1)
            Ko = K(Xr,Xo,gamma_r,gamma_s,M,br,kr,r1)
            A = (Ki-2*Kc+Ko)/(2*e**2)
            B = (Ko-Ki)/(2*e)
            C = Kc
            
            # Determine if planar or non-planar I_ij definition should be used
            if abs(zeta_0)<1e-6:
                I_ij = I_plan(A,B,C,e,eta_0)
            else:
                I_ij = I_nonplan(A,B,C,e,eta_0,zeta_0,r1)
            D[i,j]=delta_x_j*np.cos(lambda_j)/(8.*np.pi)*I_ij
            
            if symxz:
                # Calculate sweep of sending box
                lambda_j = -lambda_j
                # Calculate parameters invloved in aproximate I_ij
                Xi[1] = -Xi[1]
                Xc[1] = -Xc[1]
                Xo[1] = -Xo[1]
                # Sending box dihedral
                gamma_s = -gamma_s
                eta_0 = eta(Xr[1],Xc[1],Xr[2],Xc[2],gamma_s)
                zeta_0 = zeta(Xr[1],Xc[1],Xr[2],Xc[2],gamma_s)
                r1 = np.sqrt(eta_0**2+zeta_0**2)
                Ki = K(Xr,Xi,gamma_r,gamma_s,M,br,kr,r1)
                Kc = K(Xr,Xc,gamma_r,gamma_s,M,br,kr,r1)
                Ko = K(Xr,Xo,gamma_r,gamma_s,M,br,kr,r1)
                A = (Ki-2*Kc+Ko)/(2*e**2)
                B = (Ko-Ki)/(2*e)
                C = Kc
                # Determine if planar or non-planar I_ij definition should be used
                if abs(zeta_0)<1e-6:
                    I_ij = I_plan(A,B,C,e,eta_0)
                else:
                    I_ij = I_nonplan(A,B,C,e,eta_0,zeta_0,r1)
                D[i,j]+=delta_x_j*np.cos(lambda_j)/(8.*np.pi)*I_ij
    return D

class Airfoil:
    """Creates an airfoil object.
    
    This class creates an airfoil object. Currently this class is primarily
    used in the generation of cross-sectional meshes. Currently only NACA 4
    series arfoil and rectangular boxes are supported.
    
    :Attributes:
    
    - `c (float)`: The chord length of the airfoil.
    - `t (float)`: The max percent thickness of the airfoil.
    - `p (float)`: The location of the max camber of the airfoil, in 10%
        increments.
    - `m (float)`: The max camber of the airfoil as a percent of the chord.
        
    :Methods:
    
    - `points`: Generates the x and y upper and lower coordinates of the
        airfoil.
        
    """
    def __init__(self,c,**kwargs):
        """Airfoil object constructor.
        
        Initializes the airfoil object.
        
        :Args:
        
        - `c (float)`: The chord length of the airfoil.
        - `name (str)`: The name of the airfoil section. This can either be 
            a 'NACAXXXX' airfoil or 'box' which signifies the OML is a
            rectangle.
        
        :Returns:
        
        - None
        """
        self.c = c
        name = kwargs.pop('name','NACA0012')
        if name=='box':
            pass
        else:
            self.t = float(name[-2:])/100
            self.p = float(name[-3])/10
            self.m = float(name[-4])/100
        self.name = name
    def points(self,x):
        """Generates upper and lower airfoil curves.
        
        This method will generate the x and y coordinates for the upper and
        lower airfoil surfaces provided the non-dimensional array of points x.
        
        :Args:
        
        - `x (1xN np.array[float])`: An array of floats for which the upper and
            lower airfoil curves should be generated.
            
        :Returns:
        
        - `xu (1xN np.array[float])`: The upper x-coordinates of the curve.
        - `yu (1xN np.array[float])`: The upper y-coordinates of the curve.
        - `xl (1xN np.array[float])`: The lower x-coordinates of the curve.
        - `yl (1xN np.array[float])`: The lower y-coordinates of the curve.
        """
        #Inputs:
        #x, a non-dimensional chord length from the leading edge
        x = x*self.c
        # For more detail on the NACA 4 series airfoil,
        # see: https://en.wikipedia.org/wiki/NACA_airfoil
        #TODO: Comment this method more thuroughly
        if self.name=='box':
            return x,self.c*np.ones(len(x))/2,x,-self.c*np.ones(len(x))/2
        else:
            c = self.c
            t = self.t
            m = self.m
            p = self.p
            yt = 5*t*c*(0.2969*np.sqrt(x/c)-.126*(x/c)-.3516*(x/c)**2+.2843*(x/c)**3-.1015*(x/c)**4)
            
            xc0 = x[x<c*p]
            xc1 = x[x>=c*p]
            
            if len(xc0)>0:
                yc0 = (m*xc0/p**2)*(2*p-xc0/c)
                dyc0dx = (2*m/p**2)*(p-xc0/c)
            else:
                yc0 = []
                dyc0dx = []
            if len(xc1)>0:
                yc1 = m*((c-xc1)/(1-p)**2)*(1+xc1/c-2*p)
                dyx1dx = (2*m/(1-p)**2)*(p-xc1/c)
            else:
                yc1 = []
                dyx1dx = []
            yc = np.append(yc0,yc1)
            dycdx = np.append(dyc0dx,dyx1dx)
            
            th = np.arctan(dycdx)
            
            xu = x-yt*np.sin(th)
            yu = yc+yt*np.cos(th)
            xl = x+yt*np.sin(th)
            yl = yc-yt*np.cos(th)
            return xu,yu,xl,yl
    def printSummary(self,x):
        """A method for printing a summary of the airfoil object.
        
        Prints the airfoil chord length as well as airfoil name.
        
        :Args:
        
        - None
            
        :Returns:
        
        - (str): Prints the tabulated chord length and name of the airfoil
        
        """
        print('Airfoil name: %s' %(self.name))
        print('Airfoil Chord length: %4.4f' %(self.c))
        

class CQUADA:
    """Represents a CQUADA aerodynamic panel.
    
    This CQUADA panel object is used for the unsteady aerodynamic doublet-
    lattice method currently, although it could likely easily be extended to
    support the vortex lattice method as well. The geometry of a generic panel
    can be seen in the figure below.
    
    .. image:: images/DoubletLatticePanel.png
        :align: center
    
    :Attributes:
    
    - `type (str)`: The type of object.
    - `PANID (int)`: The integer ID linked with the panel.
    - `xs (1x4 np.array[float])`: The x coordinates of the panel.
    - `ys (1x4 np.array[float])`: The y coordinates of the panel.
    - `zs (1x4 np.array[float])`: The z coordinates of the panel.
    - `DOF (dict[NID,factor])`: This dictionary is for connecting the movement
        of the panel to the movement of an associated structure. Since a
        panel's control point could be between two nodes (in the middle of an
        element), the position of the panel can interpolated using a finite
        element formulation. The NID's link the movement of the panel to the
        movement of a corresponding node. The factor allows for a finite
        element interpolation.
    - `Area (float)`: The area of the panel.
    - `sweep (float)`: The average sweep of the panel's doublet line.
    - `delta_x (float)`: The average chord line of the panel.
    - `l (float)`: The length of the panel's doublet line.
    - `dihedral (float)`: The dihedral of the panel.
    - `Xr (1x3 np.array[float])`: The coordiantes of the panel's sending point.
    - `Xi (1x3 np.array[float])`: The coordinates of the panel's inboard
        sending point.
    - `Xc (1x3 np.array[float])`: The coordinates of the panel's center
        sending point.
    - `Xo (1x3 np.array[float])`: The coordinates of the panel's outboard
        sending point.
    
    :Methods:
    
    - `x`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the x coordinates.
    - `y`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the y coordinates.
    - `z`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the z coordinates.
    - `J`:Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns the jacobian matrix at that point. This
        method is primarily used to fascilitate the calculation of the panels
        area.
    - `printSummary`: Prints a summary of the panel.
        
    .. Note:: The ordering of the xs, ys, and zs arrays should be ordered in a
    finite element convention. The first point refers to the root trailing edge
    point, followed by the tip trailling edge, then the tip leading edge, then
    root leading edge.
    
    """
    def __init__(self,PANID,xs):
        """Initializes the panel.
        
        This method initializes the panel, including generating many of the
        geometric properties required for the doublet lattice method such as
        Xr, Xi, etc.
        
        :Args:
        
        - `PANID (int)`: The integer ID associated with the panel.
        - `xs (1x4 array[1x3 np.array[float]])`: The coordinates of the four
            corner points of the elements.
            
            
        :Returns:
        
        - None
        
        """
        # Initialize type
        self.type = 'CQUADA'
        # Error checking on EID input
        if type(PANID) is int:
            self.PANID = PANID
        else:
            raise TypeError('The element ID given was not an integer')
        if not len(xs) == 4:
            raise ValueError('A CQUAD4 element requires 4 coordinates, %d '+
                'were supplied in the nodes array' % (len(xs)))
        # Populate the NIDs array with the IDs of the nodes used by the element
        self.xs = []
        self.ys = []
        self.zs = []
        for x in xs:
            self.xs+=[x[0]]
            self.ys+=[x[1]]
            self.zs+=[x[2]]
        self.DOF = {}
        self.Area = 0
        # Initialize coordinates for Guass Quadrature Integration
        etas = np.array([-1,1])*np.sqrt(3)/3
        xis = np.array([-1,1])*np.sqrt(3)/3
        # Evaluate/sum the cross-section matricies at the Guass points
        for k in range(0,np.size(xis)):
            for l in range(0,np.size(etas)):
                Jmat = self.J(etas[l],xis[k])
                #Get determinant of the Jacobian Matrix
                Jdet = abs(np.linalg.det(Jmat))
                # Calculate the mass per unit length of the element
                self.Area += Jdet
        # Calculate box sweep angle
        xtmp = self.x(1,-1)-self.x(-1,-1)
        ytmp = self.y(1,-1)-self.y(-1,-1)
        sweep = np.arctan(xtmp/ytmp)
        if abs(sweep)<1e-3:
            sweep = 0.
        self.sweep = sweep
        # Calculate the average chord length
        self.delta_x = self.x(0,1)-self.x(0,-1)
        # Calculate the length of the doublet line
        xtmp = self.x(1,.5)-self.x(-1,.5)
        ytmp = self.y(1,.5)-self.y(-1,.5)
        ztmp = self.z(1,.5)-self.z(-1,.5)
        self.l = np.linalg.norm([xtmp,ytmp,ztmp])
        # Calculate box dihedral
        dihedral = np.arctan(ztmp/ytmp)
        if abs(dihedral)<1e-3:
            dihedral = 0.
        self.dihedral = dihedral
        # Calculate sending and recieving points on the box
        self.Xr = np.array([self.x(0.,.5),self.y(0.,.5),\
            self.z(0.,.5)])
        self.Xi = np.array([self.x(-1,-.5),self.y(-1,.5),\
            self.z(-1,-.5)])
        self.Xc = np.array([self.x(0.,-.5),self.y(0.,-.5),\
            self.z(0.,-.5)])
        self.Xo = np.array([self.x(1.,-.5),self.y(1.,-.5),\
            self.z(1.,-.5)])
        
    def x(self,eta,xi):
        """Calculate the x-coordinate within the panel.
        
        Calculates the x-coordinate on the panel provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `x (float)`: The x-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
    def y(self,eta,xi):
        """Calculate the y-coordinate within the panel.
        
        Calculates the y-coordinate on the panel provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `y (float)`: The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
    def z(self,eta,xi):
        """Calculate the z-coordinate within the panel.
        
        Calculates the z-coordinate on the panel provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `z (float)`: The z-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        zs = self.zs
        return .25*(zs[0]*(1.-xi)*(1.-eta)+zs[1]*(1.+xi)*(1.-eta)+\
                zs[2]*(1.+xi)*(1.+eta)+zs[3]*(1.-xi)*(1.+eta))
    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.
        
        This method calculates the jacobian at a local point within the panel
        provided the master coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        #TODO: Add support for panels not in x-y plane
        xs = self.xs
        ys = self.ys
        zs = self.zs
        J11 = 0.25*(-xs[0]*(1-eta)+xs[1]*(1-eta)+xs[2]*(1+eta)-xs[3]*(1+eta))
        J12 = 0.25*(-ys[0]*(1-eta)+ys[1]*(1-eta)+ys[2]*(1+eta)-ys[3]*(1+eta))
        #J13 = 0.25*(-zs[0]*(1-eta)+zs[1]*(1-eta)+zs[2]*(1+eta)-zs[3]*(1+eta))
        J21 = 0.25*(-xs[0]*(1-xi)-xs[1]*(1+xi)+xs[2]*(1+xi)+xs[3]*(1-xi))
        J22 = 0.25*(-ys[0]*(1-xi)-ys[1]*(1+xi)+ys[2]*(1+xi)+ys[3]*(1-xi))
        #J23 = 0.25*(-zs[0]*(1-xi)-zs[1]*(1+xi)+zs[2]*(1+xi)+zs[3]*(1-xi))
        # Last row of Jmat is unit normal vector of panel
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def printSummary(self):
        """A method for printing a summary of the CQUADA panel.
        
        Prints out a tabulated information about the panel such as it's panel
        ID, and the coordinates of it's four corner points.
        
        :Args:
        
        - None
            
        :Returns:
        
        - `summary (str)`: The summary of the CQUADA attributes.
            
        """
        print('CQUADA Summary:')
        print('PANID: %d' %(self.PANID))
        headers = ('Coordinates','x','y','z')
        tabmat = np.zeros((4,4),dtype=object)
        tabmat[:,0] = np.array(['Point 1','Point 2','Point 3','Point 4'])
        for i in range(len(self.xs)):
            tabmat[i,1:] = np.array([self.xs[i],self.ys[i],self.zs[i]])
        print(tabulate(tabmat,headers,tablefmt="fancy_grid"))

class CAERO1:
    """Represents an aerodynamic surface.
    
    This CAERO1 object represents an aerodynamic lifting surface to be modeled
    using the doublet-lattice method.
    
    :Attributes:
    
    - `type (str)`: The type of object.
    - `SID (int)`: The integer ID linked with the surface.
    - `xs (1x4 np.array[float])`: The x coordinates of the panel.
    - `ys (1x4 np.array[float])`: The y coordinates of the panel.
    - `zs (1x4 np.array[float])`: The z coordinates of the panel.
    - `mesh ((NPAN+1)x(NPAN+1) np.array[int])`: The panel ID's in the relative
        positions of their corresponding panels.
    - `xmesh ((NPAN+1)x(NPAN+1) np.array[float])`: The x-coordinates of the
        lifting surface nodes.
    - `ymesh ((NPAN+1)x(NPAN+1) np.array[float])`: The y-coordinates of the
        lifting surface nodes.
    - `zmesh ((NPAN+1)x(NPAN+1) np.array[float])`: The z-coordinates of the
        lifting surface nodes.
    - `CQUADAs (dict[PANID, CQUADA])`: A dictionary mapping panel ID's to
        CQUADA panel objects.
    
    :Methods:
    
    - `x`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the x coordinates.
    - `y`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the y coordinates.
    - `z`: Provided the non-dimensional coordinates eta and xi which go from -1
        to 1, this method returns corresponding the z coordinates.
    - `plotLiftingSurface`: Plots the lifting surface in 3D space. Useful for
        debugging purposes.
    - `printSummary`: Prints a summary of the panel.
        
    .. Note:: The ordering of the xs, ys, and zs arrays should be ordered in a
    finite element convention. The first point refers to the root leading edge
    point, followed by the root trailling edge, then the tip trailing edge,
    then tip leading edge.
    
    """
    def __init__(self,SID,x1,x2,x3,x4,nspan,nchord,**kwargs):
        """Constructor for the CAERO1 lifting surface object.
        
        Provided several geometric parameters, this method initializes and
        discretizes a lifting surface using CQUADA panel objects.
        
        :Args:
        
        - `SID (int)`: The integer ID for the surface.
        - `x1 (1x3 np.array[float])`: The coordinate of the root leading edge.
        - `x2 (1x3 np.array[float])`: The coordinate of the root trailing edge.
        - `x3 (1x3 np.array[float])`: The coordinate of the tip trailing edge.
        - `x4 (1x3 np.array[float])`: The coordinate of the tip leading edge.
        - `nspan (int)`: The number of panels to run in the spanwise direction.
        - `nchord (int)`: The number of panels to run in the chordwise
            direction.
            
        :Returns:
        
        - None
            
        """
        # Initialize type
        self.type = 'CAERO1'
        # Error checking on SID input
        if type(SID) is int:
            self.SID = SID
        else:
            raise TypeError('The element ID given was not an integer')
        #TODO: Thrown in a check to make sure x1 and x2 share at least one value
        #TODO: Thrown in a check to make sure x4 and x3 share at least one value
        # Starting aero box ID
        SPANID = kwargs.pop('SPANID',0)
        # Populate the NIDs array with the IDs of the nodes used by the element
        self.xs = [x1[0],x2[0],x3[0],x4[0]]
        self.ys = [x1[1],x2[1],x3[1],x4[1]]
        self.zs = [x1[2],x2[2],x3[2],x4[2]]
        
        # Generate Grids in superelement space
        xis = np.linspace(-1,1,nchord+1)
        etas = np.linspace(-1,1,nspan+1)
        
        xis, etas = np.meshgrid(xis,etas)
        
        self.xmesh = self.x(etas,xis)
        self.ymesh = self.y(etas,xis)
        self.zmesh = self.z(etas,xis)
        self.mesh = np.zeros((nchord,nspan),dtype=int)
        
        self.CQUADAs = {SPANID-1:None}
        
        
        for i in range(0,nspan):
            for j in range(0,nchord):
                newPANID = max(self.CQUADAs.keys())+1
                self.mesh[j,i] = newPANID
                x1 = [self.xmesh[i,j],self.ymesh[i,j],self.zmesh[i,j]]
                x2 = [self.xmesh[i,j+1],self.ymesh[i,j+1],self.zmesh[i,j+1]]
                x3 = [self.xmesh[i+1,j+1],self.ymesh[i+1,j+1],self.zmesh[i+1,j+1]]
                x4 = [self.xmesh[i+1,j],self.ymesh[i+1,j],self.zmesh[i+1,j]]
                self.CQUADAs[newPANID] = CQUADA(newPANID,[x1,x2,x3,x4])
        del self.CQUADAs[-1]
        
    def x(self,eta,xi):
        """Calculate the x-coordinate within the lifting surface.
        
        Calculates the x-coordinate within the lifting surface provided the
        desired master coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `x (float)`: The x-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
    def y(self,eta,xi):
        """Calculate the y-coordinate within the lifting surface.
        
        Calculates the y-coordinate within the lifting surface provided the
        desired master coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `y (float)`: The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
    def z(self,eta,xi):
        """Calculate the z-coordinate within the lifting surface.
        
        Calculates the z-coordinate within the lifting surface provided the
        desired master coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `z (float)`: The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        zs = self.zs
        return .25*(zs[0]*(1.-xi)*(1.-eta)+zs[1]*(1.+xi)*(1.-eta)+\
                zs[2]*(1.+xi)*(1.+eta)+zs[3]*(1.-xi)*(1.+eta))
    def plotLiftingSurface(self,**kwargs):
        """Plots the lifting surface using the MayaVi environment.
        
        This method plots the lifting surface using the MayaVi engine. It is
        most useful for debugging models, allowing the user to verify that the
        wing they thought they generated is actually what was generated.
        
        :Args:
        
        - `figName (str)`: The name of the figure
            
        :Returns:
        
        - `(figure)`: MayaVi Figure of the laminate.
        
        """
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        mlab.figure(figure=figName)
        mlab.mesh(self.xmesh,self.ymesh,self.zmesh,representation='wireframe',color=(0,0,0))
        mlab.mesh(self.xmesh,self.ymesh,self.zmesh)
    def printSummary(self):
        """A method for printing a summary of the CAERO1 element.
        
        Prints out the surface ID, as well as the number of chordwise and
        spanwise panels.
        
        :Args:
        
        - None
            
        :Returns:
        
        - `summary (str)`: A summary of the CAERO1 surface attributes.
            
        """
        print('CQUADA Summary:')
        print('SID' %(self.SID))
        print('Number of chordwise panels: %d' %(np.size(self.mesh,axis=1)))
        print('Number of spanwise panels: %d' %(np.size(self.mesh,axis=0)))