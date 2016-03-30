#Aerodynamics.py
# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================

# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np
from tabulate import tabulate
import mayavi.mlab as mlab
from numba import jit

# Define Constants for later sumation integration
an = np.array([.24186198,-2.7918027,24.991079,-111.59196,271.43549,-305.75288,\
    -41.18363,545.98537,-644.78155,328.72755,-64.279511])
c = 0.372
n = np.array(range(1,12))
# Define Kernel Function Here
@jit(nopython=True)
def K(Xr,Xs,gamma_r,gamma_s,M,br,kr,r1):
    # Vector pointing from sending node to recieving node
    x0 = Xr[0]-Xs[0]
    y0 = Xr[1]-Xs[1]
    z0 = Xr[2]-Xs[2]
    
    # Check if r1 is very small
    if abs(r1)<1e-6:
        if x0>0:
            return 2*np.exp(-1j*kr*x0/br)
        else:
            return 0.
    # Prandtl-Glauert Compressability Correction Factor
    beta = (1-M**2)**(0.5)
    # Reduced Frequency
    k1 = r1*kr/br
    # Another distance value?
    R = (x0**2+beta**2*r1**2)**(0.5)
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
        K1 = I1_val+M*r1*np.exp(-1j*k1*u1)/(R*(1+u1**2)**(0.5))
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
    return np.exp(-1j*kr*x0/br)*(K1*T1+K2*T2)
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
    return np.exp(-1j*k1*u1)*(1-u1/(1+u1**2)**(0.5)-1j*k1*I0(u1,k1))
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

@jit
def calcAIC(M,kr,br,delta_x_vec,sweep_vec,l_vec,dihedral_vec,Xr_vec,Xi_vec,Xc_vec,\
    Xo_vec,symxz=False):
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
    #W = np.real(Wd)+1j*kr/br*np.imag(Wd)
    #B = br/kr*Bd
    #B = Bd
    #Qaic = np.dot(B,np.dot(np.linalg.inv(D),W))
    return D

class Airfoil:
    def __init__(self,c,**kwargs):
        '''
        Inputs:
        c - the chord length of the airfoil section
        Optional arguments:
        name - the NACA 4 series airfoil name
           ex, name = 'NACA0012'
        xu - the x-coordinates of the upper curve of the airfoil
        yu - the y-coordinates of the upper curve of the airfoil
        xl - the x-coordinates of the lower curve of the airfoil
        yl - the y-coordinates of the lower curve of the airfoil
        '''
        name = kwargs.pop('name','NACA0012')
        xu = kwargs.get('xu')
        yu = kwargs.get('yu')
        xl = kwargs.get('xl')
        yl = kwargs.get('yl')
        self.c = c
        #If xu, yu, xl, yl, don't generate a NACA airfoil
        if not ((xu==None) or (yu==None) or (xl==None) or (yl==None)):
            #TODO: Finish this section, will require curve fitting module, scipy.optimize
            test=1
        elif name=='box':
            pass
        else:
            self.t = float(name[-2:])/100
            self.p = float(name[-3])/10
            self.m = float(name[-4])/100
        self.name = name
    def points(self,x):
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

class CQUADA:
    
    def __init__(self,PANID,xs,**kwargs):
        """ Initializes the element.
        
        :Args:
        - `EID (int)`: An integer identifier for the CQUAD4 element.
        - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
            used to create the element.
        - `MID (int)`: An integer refrencing the material ID used for the
            constitutive relations.
        - `matLib (obj)`: A material library object containing a dictionary
            with the material corresponding to the provided MID.
        - `xsect (bool)`: A boolean to determine whether this quad element is
            to be usedfor cross-sectional analysis. Defualt value is True.
        - `th (1x3 Array[float])`: Array containing the Euler-angles expressing
            how the element constitutive relations should be rotated from
            the material fiber frame to the global CSYS. In degrees.
            
        :Returns:
        - None
        
        .. Note:: The reference coordinate system for cross-sectional analysis is a
        local coordinate system in which the x and y axes are planer with the
        element, and the z-axis is perpendicular to the plane of the element.
        
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
        """Calculate the x-coordinate within the element.
        
        Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `float`: The x-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
    def y(self,eta,xi):
        """Calculate the y-coordinate within the element.
        
        Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `(float)': The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
    def z(self,eta,xi):
        """Calculate the z-coordinate within the element.
        
        Intended primarily as a private method but left public, this method
        calculates the transformation matrix that converts stresses to force
        and moment resultants.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `(float)': The z-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        zs = self.zs
        return .25*(zs[0]*(1.-xi)*(1.-eta)+zs[1]*(1.+xi)*(1.-eta)+\
                zs[2]*(1.+xi)*(1.+eta)+zs[3]*(1.-xi)*(1.+eta))
    def J(self,eta,xi):
        """Calculates the jacobian at a point in the element.
        
        This method calculates the jacobian at a local point within the element
        provided the master coordinates eta and xi.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `Jmat (3x3 np.array[float])`: The stress-resutlant transformation
            array.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        ys = self.ys
        J11 = 0.25*(-xs[0]*(1-eta)+xs[1]*(1-eta)+xs[2]*(1+eta)-xs[3]*(1+eta))
        J12 = 0.25*(-ys[0]*(1-eta)+ys[1]*(1-eta)+ys[2]*(1+eta)-ys[3]*(1+eta))
        J21 = 0.25*(-xs[0]*(1-xi)-xs[1]*(1+xi)+xs[2]*(1+xi)+xs[3]*(1-xi))
        J22 = 0.25*(-ys[0]*(1-xi)-ys[1]*(1+xi)+ys[2]*(1+xi)+ys[3]*(1-xi))
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def printSummary(self):
        """A method for printing a summary of the CQUAD4 element.
        
        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.
        
        :Args:
        - None
            
        :Returns:
        - (str): Prints the tabulated EID, node IDs and material IDs associated
            with the CQUAD4 element.
            
        """
        print('CQUADA Summary:')
        headers = ('EID','NID 1','NID 2','NID 3','NID 4')
        print(tabulate([[self.PANID]+self.NIDs],headers,tablefmt="fancy_grid"))

class CAERO1:
    def __init__(self,SID,x1,x2,x3,x4,nspan,nchord,**kwargs):
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
        
        self.CQUADAs = {SPANID-1:None}
        
        
        for i in range(0,nspan):
            for j in range(0,nchord):
                newPANID = max(self.CQUADAs.keys())+1
                x1 = [self.xmesh[i,j],self.ymesh[i,j],self.zmesh[i,j]]
                x2 = [self.xmesh[i,j+1],self.ymesh[i,j+1],self.zmesh[i,j+1]]
                x3 = [self.xmesh[i+1,j+1],self.ymesh[i+1,j+1],self.zmesh[i+1,j+1]]
                x4 = [self.xmesh[i+1,j],self.ymesh[i+1,j],self.zmesh[i+1,j]]
                self.CQUADAs[newPANID] = CQUADA(newPANID,[x1,x2,x3,x4])
        del self.CQUADAs[-1]
        
    def x(self,eta,xi):
        """Calculate the x-coordinate within the element.
        
        Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `float`: The x-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
    def y(self,eta,xi):
        """Calculate the y-coordinate within the element.
        
        Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `(float)': The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
    def z(self,eta,xi):
        """Calculate the z-coordinate within the element.
        
        Intended primarily as a private method but left public, this method
        calculates the transformation matrix that converts stresses to force
        and moment resultants.
        
        :Args:
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        - `(float)': The z-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        zs = self.zs
        return .25*(zs[0]*(1.-xi)*(1.-eta)+zs[1]*(1.+xi)*(1.-eta)+\
                zs[2]*(1.+xi)*(1.+eta)+zs[3]*(1.-xi)*(1.+eta))
    def plotLiftingSurface(self,**kwargs):
        """Plots the laminate using the MayaVi environment.
        
        Intended primarily as a private method but left public, this method,
        Plots a 2D representation of the laminate. This method can only be
        employed once a cross-section object has been instantiated using the
        laminate object desired. This method is outdated and generally no
        longer used.
        
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
        """A method for printing a summary of the CQUAD4 element.
        
        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.
        
        :Args:
        - None
            
        :Returns:
        - (str): Prints the tabulated EID, node IDs and material IDs associated
            with the CQUAD4 element.
            
        """
        print('CQUADA Summary:')
        headers = ('SID')
        print(tabulate([self.SID],headers,tablefmt="fancy_grid"))
