# structures.py
# Author: Ben Names
"""
This module contains a library of classes devoted to structural analysis.

The primary purpose of this library is to fascilitate the ROM (reduced order
modeling) of structures that can simplified to beams. The real power of this
library comes from it's the XSect class. This class can create and analyze
a cross-section, allowing the user to accurately model a nonhomogeneous
(made of multiple materials) anisotropic (materials that behave anisotropically
such as composites) complex cross-sections.

It should be noted that classes are ordered by model complexity. The further
down the structures.py library, the more complex the objects, often requiring
multiple of their predecessors. For example, the CQUADX class requires four
node objects and a material object.

:SUMARRY OF THE CLASSES:

- `Node`: Creates a node object with 3D position.
- `Material`: Creates a material object, generating the 3D constitutive relations.
- `MicroMechanics`: Class to fascilitate the calculation of composite stiffnesses
    using micro-mechanical models where fibers are long and continuous.
- `CQUADX`: Creates a 2D linear quadrilateral element, mainly used to fascilitate\
    cross-sectional analysis, this class could be modified in future updates
    such that they could also be used to create plate or laminate element
    objects as well.
- `MaterialLib`: Creates a material library object meant to hold many material
    objects.
- `Ply`: Creates ply objects which are used in the building of a laminate object.
- `Laminate`: Creates laminate objects which could be used for CLT (classical
    lamination theory) analysis as well as to be used in building a beam
    cross-section.
- `XSect`: Creates a cross-section object which can be used in the ROM of a beam
    with a non-homogeneous anisotropic cross-section. Currently only supports
    simple box beam cross-section (i.e., four laminates joined together to form
    a box), however outer mold lines can take the shape of airfoil profiles.
    See the Airfoil class in AircraftParts.py for more info.
- `TBeam`: Creates a single Timoshenko beam object for FEA.
- `SuperBeam`: Creates a super beam object. This class is mainly used to automate
    the creation of many connected TBeam objects to be used late for FEA.
- `WingSection`: A class which creates and holds many super beams, each of which
    could have different cross-sections. It also helps to dimensionalize
    plates for simple closed-form composite buckling load aproximations.


.. Note:: Currently the inclusion of thermal strains are not supported for any
    structural model.
    
"""
__docformat__ = 'restructuredtext'
# =============================================================================
# AeroComBAT MODULES
# =============================================================================
from Utilities import RotationHelper
from Aerodynamics import Airfoil
# =============================================================================
# IMPORT ANACONDA ASSOCIATED MODULES
# =============================================================================
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix, eye
from scipy.sparse.linalg import lgmres, minres, spsolve
from scipy import linalg
from tabulate import tabulate
#import mayavi.mlab as mlab
# =============================================================================
# IMPORT PYTHON MODULES
# =============================================================================
import collections as coll
#from memory_profiler import profile
# =============================================================================
# DEFINE AeroComBAT STRUCTURES CLASSES
# =============================================================================
class Node:
    """Creates a node object.
    
    Node objects could be used in any finite element implementation.
    
    :Attributes:
    
    - `NID (int)`: The integer identifier given to the object.
    - `x (Array[float])`: An array containing the 3 x-y-z coordinates of the
        node.
    - `summary (str)`: A string which is a tabulated respresentation and
        summary of the important attributes of the object.
        
    :Methods:
    
    - `printSummary`: This method prints out basic information about the node
        object, such as it's node ID and it's x-y-z coordinates
        
    """
    def __init__(self,NID,x):
        """Initializes the node object.
        
        :Args:
        
        - `nid (int)`: The desired integer node ID
        - `x (Array[float])`: The position of the node in 3D space.
            
        :Returns:
        
        - None
        
        """
        # Verify that a correct NID was given
        if type(NID) is int:
            self.NID = NID
        else:
            raise TypeError('The node ID given was not an integer.')
        if not len(x)==3:
            raise ValueError("An array of length 3 is required to create a "+
                "node object.")
        self.x = x
        # Initialize the summary info:
        self.summary = tabulate(([[self.NID,self.x]]),('NID','Coordinates'))
    def printSummary(self):
        """Prints basic information about the node.
        
        The printSummary method prints out basic node attributes in an organized
        fashion. This includes the node ID and x-y-z global coordinates.
        
        :Args:
        
        - None
            
        :Returns:
        
        - A printed table including the node ID and it's coordinates
            
        """
        print(self.summary)

class Material:
    """creates a linear elastic material object.
    
    This class creates a material object which can be stored within a
    material library object. The material can be in general orthotropic.
    
    :Attributes:
    
    - `name (str)`: A name for the material.
    - `MID (int)`: An integer identifier for the material.
    - `matType (str)`: A string expressing what type of material it is.
        Currently, the supported materials are isotropic, transversely
        isotropic, and orthotropic.
    - `summary (str)`: A string which is a tabulated respresentation and
        summary of the important attributes of the object.
    - `t (float)`: A single float which represents the thickness of a ply if
        the material is to be used in a composite.
    - `rho (float)`: A single float which represents the density of the
        materials.
    - `Smat (6x6 numpy Array[float])`: A numpy array representing the
        compliance matrix in the fiber coordinate system.*
    - `Cmat (6x6 numpy Array[float])`: A numpy array representing the
        stiffness matrix in the fiber coordinate system.*
        
    :Methods:
    
    - `printSummary`: This method prints out basic information about the
        material, including the type, the material constants, material
        thickness, as well as the tabulated stiffness or compliance
        matricies if requested.
        
    .. Note:: The CQUADX element assumes that the fibers are oriented along
        the (1,0,0) in the global coordinate system.
        
    """ # why is thickness defined in material and not ply?
    def __init__(self,MID,name,matType,mat_constants,mat_t,**kwargs):
        """Creates a material object
        
        The main purpose of this class is assembling the constitutive
        relations. Regardless of the analysis
            
        
        :Args:
        
        - `MID (int)`: Material ID.
        - `name (str)`: Name of the material.
        - `matType (str)`: The type of the material. Supported material types
            are "iso", "trans_iso", and "ortho".
        - `mat_constants (1xX Array[Float])`: The requisite number of material
            constants required for any structural analysis. Note, this
            array includes the material density. For example, an isotropic
            material needs 2 elastic material constants, so the total
            length of mat_constants would be 3, 2 elastic constants and the
            density.
        - `mat_t (float)`: The thickness of 1-ply of the material
        - `th (1x3 Array[float])`: The angles about which the material can be
            rotated when it is initialized. In degrees.
            
        
        :Returns:
        
        - None
            
        .. Note:: While this class supports material direction rotations, it is more
            robust to simply let the CQUADX and Mesher class handle all material
            rotations.
            
        """
        # Initialize rotation of the material in the global CSYS
        th = kwargs.pop('th', [0,0,0])
        # Initialize Material Name
        self.name = name
        # Material identification
        # Error checking to verify ID is of type int
        if type(MID) is int:
            self.MID = MID
        else:
            raise TypeError('The material ID given was not an integer') #repeats
        # Material Type(string) - isotropic, transversely isotropic, otrthotropic
        self.matType = matType
        # Material Constants(array if floats) - depends on matType
        saved_mat_const = []
        
        # ISOTROPIC MATERIAL
        if matType=='iso' and len(mat_constants)==3:
            # mat_constants expected = [E, nu, rho]
            E = mat_constants[0]
            nu = mat_constants[1]
            rho = mat_constants[2]
            G = E/(2*(1+nu))
            saved_mat_const = [E, E, E, nu, nu, nu, G, G, G, rho]
            self.summary = tabulate([[MID,'iso',E,nu,G,rho]],\
            ('MID','Type','E(Pa)','nu','G(Pa)','rho, kg/m^3'),tablefmt="fancy_grid")
        
        # TRANSVERSELY ISOTROPIC MATERIAL
        elif matType=='trans_iso' and len(mat_constants)==6:
            # mat_constants expected = [E1, E2, nu_23, nu_12, G_12, rho]
            E1 = mat_constants[0]
            E2 = mat_constants[1]
            nu_23 = mat_constants[2]
            nu_12 = mat_constants[3]
            G_12 = mat_constants[4]
            G_23 = E2/(2*(1+nu_23))
            rho = mat_constants[5]
            saved_mat_const = [E1, E2, E2, nu_23, nu_12, nu_12, G_23, G_12, G_12, rho]
            self.summary = tabulate([[MID,'trans_iso',E1,E2,nu_23,nu_12,G_23,G_12,rho]],\
            ('MID','Type','E1(Pa)','E2(Pa)','nu_23','nu_12','G_23(Pa)','G_12(Pa)',\
            'rho, kg/m^3'),tablefmt="fancy_grid")
        
        # ORTHOTROPIC MATERIAL
        elif matType=='ortho' and len(mat_constants)==10:
            # mat_constants expected = [E1,E2,E3,nu_23,nu_13,nu_12,G_23,G_13,G_12,rho]
            saved_mat_const = mat_constants #re-order
            E1 = mat_constants[0]
            E2 = mat_constants[1]
            E3 = mat_constants[2]
            nu_23 = mat_constants[3]
            nu_13 = mat_constants[4]
            nu_12 = mat_constants[5]
            G_23 = mat_constants[6]
            G_13 = mat_constants[7]
            G_12 = mat_constants[8]
            rho = mat_constants[9]
            self.summary = tabulate([[MID,'ortho',E1,E2,E3,nu_23,nu_13,nu_12,G_23,G_13,G_12,rho]],\
            ('MID','Type','E1(Pa)','E2(Pa)','E3(Pa)','nu_23','nu_13','nu_12',\
            'G_23(Pa)','G_13(Pa)','G_12(Pa)','rho, kg/m^3'),tablefmt="fancy_grid")
        else:
            raise ValueError('Material %s was not entered correctly. Possible'\
            +'material types include "iso", "trans_iso", or "ortho." In'\
            +'addition, mat_constants must then be of length 3,6, or 10'\
            +'respectively. Refer to documentation for more clarification.')
        # Store material constants such that:
        self.E1 = saved_mat_const[0]
        self.E2 = saved_mat_const[1]
        self.E3 = saved_mat_const[2]
        self.nu_23 = saved_mat_const[3]
        self.nu_13 = saved_mat_const[4]
        self.nu_12 = saved_mat_const[5]
        self.G_23 = saved_mat_const[6]
        self.G_13 = saved_mat_const[7]
        self.G_12 = saved_mat_const[8]
        self.rho = saved_mat_const[9]
        self.t = mat_t
        
        # Initialize the compliance matrix in the local fiber 123 CSYS:
        self.Smat = np.array([[1./self.E1,-self.nu_12/self.E1,-self.nu_13/self.E1,0.,0.,0.],\
                                     [-self.nu_12/self.E1,1./self.E2,-self.nu_23/self.E2,0.,0.,0.],\
                                     [-self.nu_13/self.E1,-self.nu_23/self.E2,1./self.E3,0.,0.,0.],\
                                     [0.,0.,0.,1./self.G_23,0.,0.],\
                                     [0.,0.,0.,0.,1./self.G_13,0.],\
                                     [0.,0.,0.,0.,0.,1./self.G_12]])
        # Rotate the compliance matrix to the local x-sect csys if the material
        # is to be used for cross-sectional analysis:
        self.Smat = self.returnComplMat(th)
        # Solve for the material stiffness matrix
        self.Cmat = np.linalg.inv(self.Smat)
    def printSummary(self,**kwargs):
        """Prints a tabulated summary of the material.
        
        This method prints out basic information about the
        material, including the type, the material constants, material
        thickness, as well as the tabulated stiffness or compliance
        matricies if requested.
        
        :Args:
        
        - `compliance (str)`: A boolean input to signify if the compliance
            matrix should be printed.
        - `stiffness (str)`: A boolean input to signify if the stiffness matrix
            should be printed.
            
        :Returns:
        
        - String print out containing the material name, as well as material
            constants and other defining material attributes. If requested
            this includes the material stiffness and compliance matricies.
            
        """
        # Print Name
        print(self.name)
        # Print string summary attribute
        print(self.summary)
        # Print compliance matrix if requested
        if kwargs.pop('compliance',False):
            print('COMPLIANCE MATRIX')
            print('xyz cross-section CSYS:')
            print(tabulate(self.Smat,tablefmt="fancy_grid"))
        # Print Stiffness matrix if requested
        if kwargs.pop('stiffness',False):
            print('STIFFNESS MATRIX')
            print('xyz cross-section CSYS:')
            print(tabulate(np.around(self.Cmat,decimals=4),tablefmt="fancy_grid"))
            
    def returnComplMat(self,th,**kwargs):
        """Returns the material 6x6 compliance matrix.
        
        Mainly inteded as a private method although kept public, and
        fascilitated the transformation of the compliance matrix to another
        coordinate system.
        
        :Args:
        
        - `th (1x3 Array[float])`: The angles about which the material can be
        rotated when it is initialized. In degrees.
            
        :Returns:
        
        - `Sp`: The transformed compliance matrix.
            
        """
        # Method to return the compliance matrix
        rh = RotationHelper()
        Sp = rh.transformCompl(self.Smat,th)
        return Sp
        
class MicroMechanics:
    """An class which calculates properties using micromechanical models.
    
    This method while not currently implemented can be used to calculate
    smeared material properties given the isotropic matrix and transversely
    isotropic fiber mechanical properties.
    
    """
    def genCompProp(Vf,E1f,E2f,nu12f,G12f,E_m,nu_m,rhof,rhom,**kwargs):
        """Calculates the smeared properties of a composite.
        
        Given the fiber and matrix material information, this method assists
        with calculating the smeared mechanical properties of a composite.
        The code assumes the fibers are transversely isotropic, and the matrix
        is isotropic.
        
        This class is in beta form currently and is largely unsuported. The
        methods and formula have been tested, however the class an not been
        used or implemented with any other piece in the module.
        
        :Args:
        
        - `Vf (float)`: The fiber volume fraction
        - `E1f (float)`: The fiber stiffness in the 1-direction
        - `E2f (float)`: The fiber stiffness in the 2-direction
        - `nu12f (float)`: The in-plane fiber poisson ratio
        - `G12f (float)`: The in-plane fiber shear modulus
        - `E_m (float)`: The matrix stiffness
        - `nu_m (float)`: The matrix poisson ratio
        - `rhof (float)`: The fiber density
        - `rhom (float)`: The matrix density
        - `thermal (1x3 Array[float])`: Coefficients of thermal expansion
        - `moisture (1x3 Array[float])`: Coeffiecients of moisture expansion
            
        :Returns:
        
        - An array containing the transversely isotropic material properties
            of the smeared material.
            
        """
        #thermal = [a1f, a2f, a_m]
        thermal = kwargs.pop('thermal', [0,0,0])
        #moisture = [b1f, b2f, b_m]
        moisture = kwargs.pop('moisture', [0,0,0])
        #G_m:
        G_m = E_m/(2.*(1.+nu_m))
        #E1:
        E1 = E1f*Vf+E_m*(1-Vf)
        #E2:
        E2 = 1/((1-np.sqrt(Vf))/E_m+np.sqrt(Vf)/(E2f*np.sqrt(Vf)+(1-np.sqrt(Vf))*E_m))
        #Nu_12:
        nu_12 = nu12f*Vf+nu_m*(1-Vf)
        #Nu_23 = ???
        #TODO: Implement micro-mechanical model
        nu_23 = .458
        #G_12
        G_12 = G_m*(((G_m+G12f)-Vf*(G_m-G12f))/((G_m+G12f)+Vf*(G_m-G12f)))
        #Comp Density
        rho = rhof*Vf+rhom*(1-Vf)
        #TODO: Add thermal properties to the output set
        if not thermal==[0,0,0]:
            a1f = thermal[0]
            a2f = thermal[1]
            a_m = thermal[2]
            #Alpha_1
            a1 = (E1f*a1f*Vf+E_m*a_m*(1-Vf))/(E1f*Vf+E_m*(1-Vf))
            #Alpha_2
            a2 = (a2f-(E_m/E1)*Vf*(a_m-a1f)*(1-Vf))*Vf+(a_m+(E1f/E1)*nu_m*(a_m-a1f)*Vf)*(1-Vf)
        if not moisture==[0,0,0]:
            b1f = moisture[0]
            b2f = moisture[1]
            b_m = moisture[2]
            #Beta_1
            b1 = (E1f*b1f*Vf+E_m*b_m*(1-Vf))/(E1f*Vf+E_m*(1-Vf))
            #Beta_2
            b2 = (b2f-(E_m/E1)*Vf*(b_m-b1f)*(1-Vf))*Vf+(b_m+(E1f/E1)*nu_m*(b_m-b1f)*Vf)*(1-Vf)
        return [E1, E2, nu_12, nu_23, G_12, rho]

# 2-D CQUADX class, can be used for cross-sectional analysis
class CQUADX:
    """ Creates a linear, 2D 4 node quadrilateral element object.
    
    The main purpose of this class is to assist in the cross-sectional
    analysis of a beam, however it COULD be modified to serve as an element for
    2D plate or laminate FE analysis.
    
    :Attributes:
    
    - `type (str)`: A string designating it a CQUADX element.
    - `xsect (bool)`: States whether the element is to be used in cross-
        sectional analysis.
    - `th (1x3 Array[float])`: Array containing the Euler-angles expressing how
        the element constitutive relations should be rotated from the
        material fiber frame to the global CSYS. In degrees.
    - `EID (int)`: An integer identifier for the CQUADX element.
    - `MID (int)`: An integer refrencing the material ID used for the
        constitutive relations.
    - `NIDs (1x4 Array[int])`: Contains the integer node identifiers for the
        node objects used to create the element.
    - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
        used to create the element.
    - `xs (1x4 np.array[float])`: Array containing the x-coordinates of the
        nodes used in the element
    - `ys (1x4 np.array[float])`: Array containing the y-coordinates of the
        nodes used in the element
    - `rho (float)`: Density of the material used in the element.
    - `mass (float)`: Mass per unit length (or thickness) of the element.
    - `U (12x1 np.array[float])`: This column vector contains the CQUADXs
        3 DOF (x-y-z) displacements in the local xsect CSYS due to cross-
        section warping effects.
    - `Eps (6x4 np.array[float])`: A matrix containing the 3D strain state
        within the CQUADX element.
    - `Sig (6x4 np.array[float])`: A matrix containing the 3D stress state
        within the CQUADX element.
        
    :Methods:
    
    - `x`: Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
    - `y`: Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
    - `J`: Calculates the jacobian of the element provided the desired master
        coordinates eta and xi.
    - `resetResults`: Initializes the displacement (U), strain (Eps), and
        stress (Sig) attributes of the element.
    - `getDeformed`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element warped
        displacements in the local xsect CSYS.
    - `getStressState`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element stress at four
        points. The 3D stress state is processed to return the Von-Mises
        or Maximum Principal stress state.
    - `printSummary`: Prints out a tabulated form of the element ID, as well
        as the node ID's referenced by the element.
        
    """
    def __init__(self,EID,nodes,MID,matLib,**kwargs):
        """ Initializes the element.
        
        :Args:
        
        - `EID (int)`: An integer identifier for the CQUADX element.
        - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
            used to create the element.
        - `MID (int)`: An integer refrencing the material ID used for the
            constitutive relations.
        - `matLib (obj)`: A material library object containing a dictionary
            with the material corresponding to the provided MID.
        - `xsect (bool)`: A boolean to determine whether this quad element is
            to be used for cross-sectional analysis. Defualt value is True.
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
        self.type = 'CQUADX'
        # Used for xsect analysis?
        xsect = kwargs.pop('xsect', True)
        self.xsect = xsect
        # Initialize Euler-angles for material orientation in the xsect CSYS
        th = kwargs.pop('th', [0.,0.,0.])
        self.th = th
        # Error checking on EID input
        if type(EID) is int:
            self.EID = EID
        else:
            raise TypeError('The element ID given was not an integer')
        if not len(nodes) == 4:
            raise ValueError('A CQUADX element requires 4 nodes, %d were'+
                'supplied in the nodes array' % (len(nodes)))
        nids = []
        for node in nodes:
            nids+= [node.NID]
        if not len(np.unique(nids))==4:
            raise ValueError('The node objects used to create this CQUADX '+
                'share at least 1 NID. Make sure that no repeated node '+
                'objects were used.')
        # Error checking on MID input
        if not MID in matLib.matDict.keys():
            raise KeyError('The MID provided is not linked with any materials'+
                'within the supplied material library.')
        # Initialize the warping displacement, strain and stress results
        self.resetResults()
        # Store the MID
        self.MID = MID
        # Populate the NIDs array with the IDs of the nodes used by the element
        self.NIDs = []
        self.nodes = nodes
        for node in nodes:
            self.NIDs += [node.NID]
        
        # INITIALIZE THE ELEMENT CONSTITUTIVE MATRIX:
        # Select the element material object from the material dictionary:
        material = matLib.getMat(MID)
        # Initialize the volume density of the element
        self.rho = material.rho
        # Initialize the mass per unit length (or thickness) of the element
        self.mass = 0
        # Create a rotation helper to rotate the compliance matrix:
        rh = RotationHelper()
        # Rotate the materials compliance matrix as necessary:
        Selem = rh.transformCompl(np.copy(material.Smat),th,xsect=xsect)
        # Reorder Selem for cross-sectional analysis:
        # Initialize empty compliance matrix
        Sxsect = np.zeros((6,6))
        # Initialize reorganization key
        shuff = [0,1,5,4,3,2]
        for i in range(0,6):
            for j in range(0,6):
                Sxsect[shuff[i],shuff[j]] = Selem[i,j]
        # Store the re-ordered material stiffness matrix:
        self.Q = np.linalg.inv(Sxsect)
        
        # Initialize Matricies for later use in xsect equilibrium solution:
        self.Ae = np.zeros((6,6))
        self.Re = np.zeros((12,6))
        self.Ee = np.zeros((12,12))
        self.Ce = np.zeros((12,12))
        self.Le = np.zeros((12,6))
        self.Me = np.zeros((12,12))
        # Generate X and Y coordinates of the nodes
        xs = np.zeros(4)
        ys = np.zeros(4)
        for i in range(0,4):
            tempxyz = nodes[i].x
            xs[i] = tempxyz[0]
            ys[i] = tempxyz[1]
        # Save for ease of strain calculation on strain recovery
        self.xs = xs
        self.ys = ys
        # Initialize coordinates for Guass Quadrature Integration
        etas = np.array([-1,1])*np.sqrt(3)/3
        xis = np.array([-1,1])*np.sqrt(3)/3
        # Evaluate/sum the cross-section matricies at the Guass points
        for k in range(0,np.size(xis)):
            for l in range(0,np.size(etas)):
                #Get Z Matrix
                Zmat = self.Z(etas[l],xis[k])
                #Get BN Matricies
                Jmat = self.J(etas[l],xis[k])
                #Get determinant of the Jacobian Matrix
                Jdet = abs(np.linalg.det(Jmat))
                Jmatinv = np.linalg.inv(Jmat)
                Bxi = np.zeros((6,3))
                Beta = np.zeros((6,3))
                Bxi[0,0] = Bxi[2,1] = Bxi[3,2] = Jmatinv[0,0]
                Bxi[1,1] = Bxi[2,0] = Bxi[4,2] = Jmatinv[1,0]
                Beta[0,0] = Beta[2,1] = Beta[3,2] = Jmatinv[0,1]
                Beta[1,1] = Beta[2,0] = Beta[4,2] = Jmatinv[1,1]
                BN = np.dot(Bxi,self.dNdxi(etas[l])) + np.dot(Beta,self.dNdeta(xis[k]))
                
                #Get a few last minute matricies
                S = np.zeros((6,3));S[3,0]=1;S[4,1]=1;S[5,2]=1
                SZ = np.dot(S,Zmat)
                Nmat = self.N(etas[l],xis[k])
                SN = np.dot(S,Nmat)
                
                # Calculate the mass per unit length of the element
                self.mass += self.rho*Jdet
                
                #Add to Ae Matrix
                self.Ae += np.dot(SZ.T,np.dot(self.Q,SZ))*Jdet
                #Add to Re Matrix
                self.Re += np.dot(BN.T,np.dot(self.Q,SZ))*Jdet
                #Add to Ee Matrix
                self.Ee += np.dot(BN.T,np.dot(self.Q,BN))*Jdet
                #Add to Ce Matrix
                self.Ce += np.dot(BN.T,np.dot(self.Q,SN))*Jdet
                #Add to Le Matrix
                self.Le += np.dot(SN.T,np.dot(self.Q,SZ))*Jdet
                #Add to Me Matrix
                self.Me += np.dot(SN.T,np.dot(self.Q,SN))*Jdet
    def x(self,eta,xi):
        """Calculate the x-coordinate within the element.
        
        Calculates the local xsect x-coordinate provided the desired master
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
        """Calculate the y-coordinate within the element.
        
        Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `y (float)': The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
    def Z(self,eta,xi):
        """Calculates transformation matrix relating stress to force-moments.
        
        Intended primarily as a private method but left public, this method
        calculates the transformation matrix that converts stresses to force
        and moment resultants.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `Z (3x6 np.array[float])`: The stress-resutlant transformation array.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        return np.array([[1.,0,0,0,0,-self.y(eta,xi)],\
                         [0,1.,0,0,0,self.x(eta,xi)],\
                         [0,0,1.,self.y(eta,xi),-self.x(eta,xi),0]])
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
    def N(self,eta,xi):
        """Generates the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        Nmat = np.zeros([3,12])
        N1 = .25*(1.-xi)*(1.-eta)
        N2 = .25*(1.+xi)*(1.-eta)
        N3 = .25*(1.+xi)*(1.+eta)
        N4 = .25*(1.-xi)*(1.+eta)
        Nmat[0,0] = Nmat[1,1] = Nmat[2,2] = N1
        Nmat[0,3] = Nmat[1,4] = Nmat[2,5] = N2
        Nmat[0,6] = Nmat[1,7] = Nmat[2,8] = N3
        Nmat[0,9] = Nmat[1,10] = Nmat[2,11] = N4
        return Nmat
    def dNdxi(self,eta):
        """Generates a gradient of the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        dNdxi_mat = np.zeros([3,12])
        dN1dxi = -.25*(1-eta)
        dN2dxi = .25*(1-eta)
        dN3dxi = .25*(1+eta)
        dN4dxi = -.25*(1+eta)
        dNdxi_mat[0,0] = dNdxi_mat[1,1] = dNdxi_mat[2,2] = dN1dxi
        dNdxi_mat[0,3] = dNdxi_mat[1,4] = dNdxi_mat[2,5] = dN2dxi
        dNdxi_mat[0,6] = dNdxi_mat[1,7] = dNdxi_mat[2,8] = dN3dxi
        dNdxi_mat[0,9] = dNdxi_mat[1,10] = dNdxi_mat[2,11] = dN4dxi
        return dNdxi_mat
    def dNdeta(self,xi):
        """Generates a gradient of the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        dNdeta_mat = np.zeros([3,12])
        dN1deta = -.25*(1-xi)
        dN2deta = -.25*(1+xi)
        dN3deta = .25*(1+xi)
        dN4deta = .25*(1-xi)
        dNdeta_mat[0,0] = dNdeta_mat[1,1] = dNdeta_mat[2,2] = dN1deta
        dNdeta_mat[0,3] = dNdeta_mat[1,4] = dNdeta_mat[2,5] = dN2deta
        dNdeta_mat[0,6] = dNdeta_mat[1,7] = dNdeta_mat[2,8] = dN3deta
        dNdeta_mat[0,9] = dNdeta_mat[1,10] = dNdeta_mat[2,11] = dN4deta
        return dNdeta_mat
    def resetResults(self):
        """Resets stress, strain and warping displacement results.
        
        Method is mainly intended to prevent results for one analysis or
        sampling location in the matrix to effect the results in another.
        
        :Args:
        
        - None
            
        :Returns:
        
        - None
            
        """
        # Initialize array for element warping displacement results
        self.U = np.zeros((12,1))
        # Initialize strain vectors
        self.Eps = np.zeros((6,4))
        # Initialize stress vectors
        self.Sig = np.zeros((6,4))
    def getDeformed(self,**kwargs):
        """Returns the warping displacement of the element.
        
        Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element warped
        displacements in the local xsect CSYS.
        
        :Args:
        
        - `warpScale (float)`: A multiplicative scaling factor intended to
            exagerate the warping displacement within the cross-section.
            
        :Returns:
        
        - `xdef (2x2 np.array[float])`: warped x-coordinates at the four corner
            points.
        - `ydef (2x2 np.array[float])`: warped y-coordinates at the four corner
            points.
        - `zdef (2x2 np.array[float])`: warped z-coordinates at the four corner
            points.
            
        """
        # Initialize the scaled value of the 
        warpScale = kwargs.pop('warpScale',1)
        # Initialize the full warping displacement vector
        utmp = self.U
        # Initialize the local node ordering
        nodeInd = np.array([[3,2],[0,1]])
        # Initialize the blank x-displacement 2x2 array
        xdef = np.zeros((2,2))
        # Initialize the blank y-displacement 2x2 array
        ydef = np.zeros((2,2))
        # Initialize the blank z-displacement 2x2 array
        zdef = np.zeros((2,2))
        # For all four points
        for i in range(0,2):
            for j in range(0,2):
                # Determine the local node index
                tmpInd = nodeInd[i,j]
                # Determine the rigid coordinate at the point
                x0 = self.nodes[tmpInd].x
                # Add the warping displacement and the rigid coordinate
                xdef[i,j] = warpScale*utmp[3*tmpInd,0]+x0[0]
                ydef[i,j] = warpScale*utmp[3*tmpInd+1,0]+x0[1]
                zdef[i,j] = warpScale*utmp[3*tmpInd+2,0]+x0[2]
        # Return the 3 displacement arrays
        return xdef,ydef,zdef
    
    def getStressState(self,crit='VonMis'):
        """Returns the stress state of the element.
        
        Provided an analysis has been conducted, this method
        returns a 2x2 np.array[float] containing the element the 3D stress
        state at the four guass points by default.*
        
        :Args:
        
        - `crit (str)`: Determines what criteria is used to evaluate the 3D
            stress state at the sample points within the element. By
            default the Von Mises stress is returned. Currently supported
            options include: Von Mises ('VonMis'), maximum principle stress
            ('MaxPrin'), the minimum principle stress ('MinPrin'), and the
            local cross-section stress states 'sig_xx' where the subindeces can
            go from 1-3. The keyword 'none' is also an option.
            
        :Returns:
        
        - `sigData (2x2 np.array[float])`: The stress state evaluated at four
            points within the CQUADX element.
            
        .. Note:: The XSect method calcWarpEffects is what determines where strain
        and stresses are sampled. By default it samples this information at the
        Guass points where the stress/strain will be most accurate.
        
        """
        #TODO: Make this method the plotting method so that stress can be
        # written to a file using another method.
        #TODO: Add routines for the determining composite failure criteria
        # such as the Tsai-Wu, Hoffman, or Puc criteria
        # Initialize the last saved 3D stress state
        sigState = self.Sig
        # Initialize the local node ordering
        nodeInd = np.array([[3,2],[0,1]])
        # Initialize the blank stress 2x2 array
        sigData = np.zeros((2,2))
        # For all four points
        for i in range(0,2):
            for j in range(0,2):
                # Determine the local node index
                tmpInd = nodeInd[i,j]
                # Determine what criteria is to be used to evaluate the stress
                # State
                if crit=='VonMis':
                    sigData[i,j] = np.sqrt(0.5*((sigState[0,tmpInd]-sigState[1,tmpInd])**2+\
                        (sigState[1,tmpInd]-sigState[5,tmpInd])**2+\
                        (sigState[5,tmpInd]-sigState[0,tmpInd])**2+\
                        6*(sigState[2,tmpInd]**2+sigState[3,tmpInd]**2+sigState[4,tmpInd]**2)))
                elif crit=='MaxPrin':
                    tmpSig = sigState[:,tmpInd]
                    tmpSigTens = np.array([[tmpSig[0],tmpSig[2],tmpSig[3]],\
                        [tmpSig[2],tmpSig[1],tmpSig[4]],\
                        [tmpSig[3],tmpSig[4],tmpSig[5]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    sigData[i,j] = max(eigs)
                elif crit=='MinPrin':
                    tmpSig = sigState[:,tmpInd]
                    tmpSigTens = np.array([[tmpSig[0],tmpSig[2],tmpSig[3]],\
                        [tmpSig[2],tmpSig[1],tmpSig[4]],\
                        [tmpSig[3],tmpSig[4],tmpSig[5]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    sigData[i,j] = min(eigs)
                elif crit=='sig_11':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[0]
                elif crit=='sig_22':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[1]
                elif crit=='sig_12':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[2]
                elif crit=='sig_13':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[3]
                elif crit=='sig_23':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[4]
                elif crit=='sig_33':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[5]
                elif crit=='none':
                    sigData[i,j] = 0.
                elif crit=='rss_shear':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = np.sqrt(tmpSig[3]**2+tmpSig[4]**2)
                
                    
        return sigData
    
    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.
        
        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.
        
        :Args:
        
        - None
            
        :Returns:
        
        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.
            
        """
        print('CQUADX Summary:')
        headers = ('EID','NID 1','NID 2','NID 3','NID 4','MID')
        print(tabulate([[self.EID]+self.NIDs+[self.MID]],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()
    def clearXSectionMatricies(self):
        """Clears large matricies associated with cross-sectional analaysis.
        
        Intended primarily as a private method but left public, this method
        clears the matricies associated with cross-sectional analysis. This is
        mainly done as a way of saving memory.
        
        """
        self.Ae = None
        self.Ce = None
        self.Ee = None
        self.Le = None
        self.Me = None
        self.Re = None
# 2-D CQUADX class, can be used for cross-sectional analysis
class CQUADX9:
    """ Creates a linear, 2D 4 node quadrilateral element object.
    
    The main purpose of this class is to assist in the cross-sectional
    analysis of a beam, however it COULD be modified to serve as an element for
    2D plate or laminate FE analysis.
    
    :Attributes:
    
    - `type (str)`: A string designating it a CQUADX element.
    - `xsect (bool)`: States whether the element is to be used in cross-
        sectional analysis.
    - `th (1x3 Array[float])`: Array containing the Euler-angles expressing how
        the element constitutive relations should be rotated from the
        material fiber frame to the global CSYS. In degrees.
    - `EID (int)`: An integer identifier for the CQUADX element.
    - `MID (int)`: An integer refrencing the material ID used for the
        constitutive relations.
    - `NIDs (1x4 Array[int])`: Contains the integer node identifiers for the
        node objects used to create the element.
    - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
        used to create the element.
    - `xs (1x4 np.array[float])`: Array containing the x-coordinates of the
        nodes used in the element
    - `ys (1x4 np.array[float])`: Array containing the y-coordinates of the
        nodes used in the element
    - `rho (float)`: Density of the material used in the element.
    - `mass (float)`: Mass per unit length (or thickness) of the element.
    - `U (12x1 np.array[float])`: This column vector contains the CQUADXs
        3 DOF (x-y-z) displacements in the local xsect CSYS due to cross-
        section warping effects.
    - `Eps (6x4 np.array[float])`: A matrix containing the 3D strain state
        within the CQUADX element.
    - `Sig (6x4 np.array[float])`: A matrix containing the 3D stress state
        within the CQUADX element.
        
    :Methods:
    
    - `x`: Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
    - `y`: Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
    - `J`: Calculates the jacobian of the element provided the desired master
        coordinates eta and xi.
    - `resetResults`: Initializes the displacement (U), strain (Eps), and
        stress (Sig) attributes of the element.
    - `getDeformed`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element warped
        displacements in the local xsect CSYS.
    - `getStressState`: Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element stress at four
        points. The 3D stress state is processed to return the Von-Mises
        or Maximum Principal stress state.
    - `printSummary`: Prints out a tabulated form of the element ID, as well
        as the node ID's referenced by the element.
        
    """
    def __init__(self,EID,nodes,MID,matLib,**kwargs):
        """ Initializes the element.
        
        :Args:
        
        - `EID (int)`: An integer identifier for the CQUADX element.
        - `nodes (1x4 Array[obj])`: Contains the properly ordered nodes objects
            used to create the element.
        - `MID (int)`: An integer refrencing the material ID used for the
            constitutive relations.
        - `matLib (obj)`: A material library object containing a dictionary
            with the material corresponding to the provided MID.
        - `xsect (bool)`: A boolean to determine whether this quad element is
            to be used for cross-sectional analysis. Defualt value is True.
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
        self.type = 'CQUADX9'
        # Used for xsect analysis?
        xsect = kwargs.pop('xsect', True)
        self.xsect = xsect
        # Initialize Euler-angles for material orientation in the xsect CSYS
        th = kwargs.pop('th', [0.,0.,0.])
        self.th = th
        # Error checking on EID input
        if type(EID) is int:
            self.EID = EID
        else:
            raise TypeError('The element ID given was not an integer')
        if not len(nodes) == 9:
            raise ValueError('A CQUADX element requires 9 nodes, %d were'+
                'supplied in the nodes array' % (len(nodes)))
        nids = []
        for node in nodes:
            nids+= [node.NID]
        if not len(np.unique(nids))==9:
            raise ValueError('The node objects used to create this CQUADX '+
                'share at least 1 NID. Make sure that no repeated node '+
                'objects were used.')
        # Error checking on MID input
        if not MID in matLib.matDict.keys():
            raise KeyError('The MID provided is not linked with any materials'+
                'within the supplied material library.')
        # Initialize the warping displacement, strain and stress results
        self.resetResults()
        # Store the MID
        self.MID = MID
        # Populate the NIDs array with the IDs of the nodes used by the element
        self.NIDs = []
        self.nodes = nodes
        for node in nodes:
            self.NIDs += [node.NID]
        
        # INITIALIZE THE ELEMENT CONSTITUTIVE MATRIX:
        # Select the element material object from the material dictionary:
        material = matLib.getMat(MID)
        # Initialize the volume density of the element
        self.rho = material.rho
        # Initialize the mass per unit length (or thickness) of the element
        self.mass = 0
        # Create a rotation helper to rotate the compliance matrix:
        rh = RotationHelper()
        # Rotate the materials compliance matrix as necessary:
        Selem = rh.transformCompl(np.copy(material.Smat),th,xsect=xsect)
        # Reorder Selem for cross-sectional analysis:
        # Initialize empty compliance matrix
        Sxsect = np.zeros((6,6))
        # Initialize reorganization key
        shuff = [0,1,5,4,3,2]
        for i in range(0,6):
            for j in range(0,6):
                Sxsect[shuff[i],shuff[j]] = Selem[i,j]
        # Store the re-ordered material stiffness matrix:
        self.Q = np.linalg.inv(Sxsect)
        
        # Initialize Matricies for later use in xsect equilibrium solution:
        self.Ae = np.zeros((6,6))
        self.Re = np.zeros((27,6))
        self.Ee = np.zeros((27,27))
        self.Ce = np.zeros((27,27))
        self.Le = np.zeros((27,6))
        self.Me = np.zeros((27,27))
        # Generate X and Y coordinates of the nodes
        xs = np.zeros(9)
        ys = np.zeros(9)
        for i in range(len(nodes)):
            tempxyz = nodes[i].x
            xs[i] = tempxyz[0]
            ys[i] = tempxyz[1]
        # Save for ease of strain calculation on strain recovery
        self.xs = xs
        self.ys = ys
        # Initialize coordinates for Guass Quadrature Integration
        etas = np.array([-1,0,1])*np.sqrt(3./5)
        xis = np.array([-1,0,1])*np.sqrt(3./5)
        w_etas = np.array([5./9,8./9,5./9])
        w_xis = np.array([5./9,8./9,5./9])
        # Evaluate/sum the cross-section matricies at the Guass points
        for xi, w_xi in zip(xis,w_xis):
            for eta, w_eta in zip(etas,w_etas):
                #Get Z Matrix
                Zmat = self.Z(eta,xi)
                #Get BN Matricies
                Jmat = self.J(eta,xi)
                #Get determinant of the Jacobian Matrix
                Jdet = abs(np.linalg.det(Jmat))
                Jmatinv = np.linalg.inv(Jmat)
                Bxi = np.zeros((6,3))
                Beta = np.zeros((6,3))
                Bxi[0,0] = Bxi[2,1] = Bxi[3,2] = Jmatinv[0,0]
                Bxi[1,1] = Bxi[2,0] = Bxi[4,2] = Jmatinv[1,0]
                Beta[0,0] = Beta[2,1] = Beta[3,2] = Jmatinv[0,1]
                Beta[1,1] = Beta[2,0] = Beta[4,2] = Jmatinv[1,1]
                BN = np.dot(Bxi,self.dNdxi(eta,xi)) + np.dot(Beta,self.dNdeta(eta,xi))
                
                #Get a few last minute matricies
                S = np.zeros((6,3));S[3,0]=1;S[4,1]=1;S[5,2]=1
                SZ = np.dot(S,Zmat)
                Nmat = self.N(eta,xi)
                #print(xi)
                #print(eta)
                #print(tabulate(Nmat))
                SN = np.dot(S,Nmat)
                
                
                # Calculate the mass per unit length of the element
                self.mass += self.rho*Jdet*w_xi*w_eta
                
                #Add to Ae Matrix
                self.Ae += np.dot(SZ.T,np.dot(self.Q,SZ))*Jdet*w_xi*w_eta
                #Add to Re Matrix
                self.Re += np.dot(BN.T,np.dot(self.Q,SZ))*Jdet*w_xi*w_eta
                #Add to Ee Matrix
                self.Ee += np.dot(BN.T,np.dot(self.Q,BN))*Jdet*w_xi*w_eta
                #Add to Ce Matrix
                self.Ce += np.dot(BN.T,np.dot(self.Q,SN))*Jdet*w_xi*w_eta
                #Add to Le Matrix
                self.Le += np.dot(SN.T,np.dot(self.Q,SZ))*Jdet*w_xi*w_eta
                #Add to Me Matrix
                self.Me += np.dot(SN.T,np.dot(self.Q,SN))*Jdet*w_xi*w_eta
    def x(self,eta,xi):
        """Calculate the x-coordinate within the element.
        
        Calculates the local xsect x-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `x (float)`: The x-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        xs = self.xs
        N = np.zeros(9)
        N[0] = .25*(xi**2-xi)*(eta**2-eta)
        N[1] = .5*(1-xi**2)*(eta**2-eta)
        N[2] = .25*(xi**2+xi)*(eta**2-eta)
        N[3] = .5*(xi**2-xi)*(1-eta**2)
        N[4] = (1-xi**2)*(1-eta**2)
        N[5] = .5*(xi**2+xi)*(1-eta**2)
        N[6] = .25*(xi**2-xi)*(eta**2+eta)
        N[7] = .5*(1-xi**2)*(eta**2+eta)
        N[8] = .25*(xi**2+xi)*(eta**2+eta)
        return np.dot(N,xs)
    def y(self,eta,xi):
        """Calculate the y-coordinate within the element.
        
        Calculates the local xsect y-coordinate provided the desired master
        coordinates eta and xi.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `y (float)': The y-coordinate within the element.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        ys = self.ys
        N = np.zeros(9)
        N[0] = .25*(xi**2-xi)*(eta**2-eta)
        N[1] = .5*(1-xi**2)*(eta**2-eta)
        N[2] = .25*(xi**2+xi)*(eta**2-eta)
        N[3] = .5*(xi**2-xi)*(1-eta**2)
        N[4] = (1-xi**2)*(1-eta**2)
        N[5] = .5*(xi**2+xi)*(1-eta**2)
        N[6] = .25*(xi**2-xi)*(eta**2+eta)
        N[7] = .5*(1-xi**2)*(eta**2+eta)
        N[8] = .25*(xi**2+xi)*(eta**2+eta)
        return np.dot(N,ys)
    def Z(self,eta,xi):
        """Calculates transformation matrix relating stress to force-moments.
        
        Intended primarily as a private method but left public, this method
        calculates the transformation matrix that converts stresses to force
        and moment resultants.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `Z (3x6 np.array[float])`: The stress-resutlant transformation array.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        return np.array([[1.,0,0,0,0,-self.y(eta,xi)],\
                         [0,1.,0,0,0,self.x(eta,xi)],\
                         [0,0,1.,self.y(eta,xi),-self.x(eta,xi),0]])
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
        # DN/Dxi
        dNdxi = np.zeros(9)
        dNdxi[0] = .25*(eta**2-eta)*(2*xi-1)
        dNdxi[1] = -(eta**2-eta)*xi
        dNdxi[2] = .25*(eta**2-eta)*(2*xi+1)
        dNdxi[3] = .5*(1-eta**2)*(2*xi-1)
        dNdxi[4] = -2*(1-eta**2)*xi
        dNdxi[5] = .5*(1-eta**2)*(2*xi+1)
        dNdxi[6] = .25*(eta**2+eta)*(2*xi-1)
        dNdxi[7] = -(eta**2+eta)*xi
        dNdxi[8] = .25*(eta**2+eta)*(2*xi+1)
        # DN/Deta
        dNdeta = np.zeros(9)
        dNdeta[0] = .25*(xi**2-xi)*(2*eta-1)
        dNdeta[1] = .5*(2*eta-1)*(1-xi**2)
        dNdeta[2] = .25*(2*eta-1)*(xi**2+xi)
        dNdeta[3] = -eta*(xi**2-xi)
        dNdeta[4] = -2*eta*(1-xi**2)
        dNdeta[5] = -eta*(xi**2+xi)
        dNdeta[6] = .25*(1+2*eta)*(xi**2-xi)
        dNdeta[7] = .5*(2*eta+1)*(1-xi**2)
        dNdeta[8] = .25*(1+2*eta)*(xi**2+xi)
        
        J11 = np.dot(dNdxi,xs)
        J12 = np.dot(dNdxi,ys)
        J21 = np.dot(dNdeta,xs)
        J22 = np.dot(dNdeta,ys)
        Jmat = np.array([[J11,J12,0],[J21,J22,0],[0,0,1]])
        return Jmat
    def N(self,eta,xi):
        """Generates the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the weighting matrix used to interpolate values within the
        element. This method however is mainly reserved for the cross-sectional
        analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `Nmat (3x12 np.array[float])`: The shape-function value weighting
            matrix.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        Nmat = np.zeros((3,3*9))
        N1 = .25*(xi**2-xi)*(eta**2-eta)
        N2 = .5*(1-xi**2)*(eta**2-eta)
        N3 = .25*(xi**2+xi)*(eta**2-eta)
        N4 = .5*(xi**2-xi)*(1-eta**2)
        N5 = (1-xi**2)*(1-eta**2)
        N6 = .5*(xi**2+xi)*(1-eta**2)
        N7 = .25*(xi**2-xi)*(eta**2+eta)
        N8 = .5*(1-xi**2)*(eta**2+eta)
        N9 = .25*(xi**2+xi)*(eta**2+eta)
        I3 = np.eye(3)
        Nmat[0:3,0:3] = N1*I3
        Nmat[0:3,3:6] = N2*I3
        Nmat[0:3,6:9] = N3*I3
        Nmat[0:3,9:12] = N4*I3
        Nmat[0:3,12:15] = N5*I3
        Nmat[0:3,15:18] = N6*I3
        Nmat[0:3,18:21] = N7*I3
        Nmat[0:3,21:24] = N8*I3
        Nmat[0:3,24:27] = N9*I3
        return Nmat
    def dNdxi(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to xi and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `dNdxi_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to xi.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        dNdxi_mat = np.zeros((3,3*9))
        # DN/Dxi
        dNdxi1 = .25*(eta**2-eta)*(2*xi-1)
        dNdxi2 = -(eta**2-eta)*xi
        dNdxi3 = .25*(eta**2-eta)*(2*xi+1)
        dNdxi4 = .5*(1-eta**2)*(2*xi-1)
        dNdxi5 = -2*(1-eta**2)*xi
        dNdxi6 = .5*(1-eta**2)*(2*xi+1)
        dNdxi7 = .25*(eta**2+eta)*(2*xi-1)
        dNdxi8 = -(eta**2+eta)*xi
        dNdxi9 = .25*(eta**2+eta)*(2*xi+1)
        I3 = np.eye(3)
        dNdxi_mat[0:3,0:3] = dNdxi1*I3
        dNdxi_mat[0:3,3:6] = dNdxi2*I3
        dNdxi_mat[0:3,6:9] = dNdxi3*I3
        dNdxi_mat[0:3,9:12] = dNdxi4*I3
        dNdxi_mat[0:3,12:15] = dNdxi5*I3
        dNdxi_mat[0:3,15:18] = dNdxi6*I3
        dNdxi_mat[0:3,18:21] = dNdxi7*I3
        dNdxi_mat[0:3,21:24] = dNdxi8*I3
        dNdxi_mat[0:3,24:27] = dNdxi9*I3
        return dNdxi_mat
    def dNdeta(self,eta,xi):
        """Generates a gradient of the shape-function value weighting matrix.
        
        Intended primarily as a private method but left public, this method
        generates the gradient of the weighting matrix with respect to eta and
        is used to interpolate values within the element. This method however
        is mainly reserved for the cross-sectional analysis process.
        
        :Args:
        
        - `eta (float)`: The eta coordinate in the master coordinate domain.*
        - `xi (float)`: The xi coordinate in the master coordinate domain.*
            
        :Returns:
        
        - `dNdeta_mat (3x12 np.array[float])`: The gradient of the shape-
            function value weighting matrix with respect to eta.
            
        .. Note:: Xi and eta can both vary between -1 and 1 respectively.
        
        """
        dNdeta_mat = np.zeros((3,3*9))
        # DN/Deta
        dNdeta1 = .25*(xi**2-xi)*(2*eta-1)
        dNdeta2 = .5*(2*eta-1)*(1-xi**2)
        dNdeta3 = .25*(2*eta-1)*(xi**2+xi)
        dNdeta4 = -eta*(xi**2-xi)
        dNdeta5 = -2*eta*(1-xi**2)
        dNdeta6 = -eta*(xi**2+xi)
        dNdeta7 = .25*(1+2*eta)*(xi**2-xi)
        dNdeta8 = .5*(2*eta+1)*(1-xi**2)
        dNdeta9 = .25*(1+2*eta)*(xi**2+xi)
        I3 = np.eye(3)
        dNdeta_mat[0:3,0:3] = dNdeta1*I3
        dNdeta_mat[0:3,3:6] = dNdeta2*I3
        dNdeta_mat[0:3,6:9] = dNdeta3*I3
        dNdeta_mat[0:3,9:12] = dNdeta4*I3
        dNdeta_mat[0:3,12:15] = dNdeta5*I3
        dNdeta_mat[0:3,15:18] = dNdeta6*I3
        dNdeta_mat[0:3,18:21] = dNdeta7*I3
        dNdeta_mat[0:3,21:24] = dNdeta8*I3
        dNdeta_mat[0:3,24:27] = dNdeta9*I3
        return dNdeta_mat
    def resetResults(self):
        """Resets stress, strain and warping displacement results.
        
        Method is mainly intended to prevent results for one analysis or
        sampling location in the matrix to effect the results in another.
        
        :Args:
        
        - None
            
        :Returns:
        
        - None
            
        """
        # Initialize array for element warping displacement results
        self.U = np.zeros((27,1))
        # Initialize strain vectors
        self.Eps = np.zeros((6,9))
        # Initialize stress vectors
        self.Sig = np.zeros((6,9))
    def getDeformed(self,**kwargs):
        """Returns the warping displacement of the element.
        
        Provided an analysis has been conducted, this method
        returns 3 2x2 np.array[float] containing the element warped
        displacements in the local xsect CSYS.
        
        :Args:
        
        - `warpScale (float)`: A multiplicative scaling factor intended to
            exagerate the warping displacement within the cross-section.
            
        :Returns:
        
        - `xdef (2x2 np.array[float])`: warped x-coordinates at the four corner
            points.
        - `ydef (2x2 np.array[float])`: warped y-coordinates at the four corner
            points.
        - `zdef (2x2 np.array[float])`: warped z-coordinates at the four corner
            points.
            
        """
        # Initialize the scaled value of the 
        warpScale = kwargs.pop('warpScale',1)
        # Initialize the full warping displacement vector
        utmp = self.U
        # Initialize the local node ordering
        nodeInd = np.array([[3,2],[0,1]])
        # Initialize the blank x-displacement 2x2 array
        xdef = np.zeros((2,2))
        # Initialize the blank y-displacement 2x2 array
        ydef = np.zeros((2,2))
        # Initialize the blank z-displacement 2x2 array
        zdef = np.zeros((2,2))
        # For all four points
        for i in range(0,2):
            for j in range(0,2):
                # Determine the local node index
                tmpInd = nodeInd[i,j]
                # Determine the rigid coordinate at the point
                x0 = self.nodes[tmpInd].x
                # Add the warping displacement and the rigid coordinate
                xdef[i,j] = warpScale*utmp[3*tmpInd,0]+x0[0]
                ydef[i,j] = warpScale*utmp[3*tmpInd+1,0]+x0[1]
                zdef[i,j] = warpScale*utmp[3*tmpInd+2,0]+x0[2]
        # Return the 3 displacement arrays
        return xdef,ydef,zdef
    
    def getStressState(self,crit='VonMis'):
        """Returns the stress state of the element.
        
        Provided an analysis has been conducted, this method
        returns a 2x2 np.array[float] containing the element the 3D stress
        state at the four guass points by default.*
        
        :Args:
        
        - `crit (str)`: Determines what criteria is used to evaluate the 3D
            stress state at the sample points within the element. By
            default the Von Mises stress is returned. Currently supported
            options include: Von Mises ('VonMis'), maximum principle stress
            ('MaxPrin'), the minimum principle stress ('MinPrin'), and the
            local cross-section stress states 'sig_xx' where the subindeces can
            go from 1-3. The keyword 'none' is also an option.
            
        :Returns:
        
        - `sigData (2x2 np.array[float])`: The stress state evaluated at four
            points within the CQUADX element.
            
        .. Note:: The XSect method calcWarpEffects is what determines where strain
        and stresses are sampled. By default it samples this information at the
        Guass points where the stress/strain will be most accurate.
        
        """
        #TODO: Make this method the plotting method so that stress can be
        # written to a file using another method.
        #TODO: Add routines for the determining composite failure criteria
        # such as the Tsai-Wu, Hoffman, or Puc criteria
        # Initialize the last saved 3D stress state
        sigState = self.Sig
        # Initialize the local node ordering
        nodeInd = np.array([[3,2],[0,1]])
        # Initialize the blank stress 2x2 array
        sigData = np.zeros((2,2))
        # For all four points
        for i in range(0,2):
            for j in range(0,2):
                # Determine the local node index
                tmpInd = nodeInd[i,j]
                # Determine what criteria is to be used to evaluate the stress
                # State
                if crit=='VonMis':
                    sigData[i,j] = np.sqrt(0.5*((sigState[0,tmpInd]-sigState[1,tmpInd])**2+\
                        (sigState[1,tmpInd]-sigState[5,tmpInd])**2+\
                        (sigState[5,tmpInd]-sigState[0,tmpInd])**2+\
                        6*(sigState[2,tmpInd]**2+sigState[3,tmpInd]**2+sigState[4,tmpInd]**2)))
                elif crit=='MaxPrin':
                    tmpSig = sigState[:,tmpInd]
                    tmpSigTens = np.array([[tmpSig[0],tmpSig[2],tmpSig[3]],\
                        [tmpSig[2],tmpSig[1],tmpSig[4]],\
                        [tmpSig[3],tmpSig[4],tmpSig[5]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    sigData[i,j] = max(eigs)
                elif crit=='MinPrin':
                    tmpSig = sigState[:,tmpInd]
                    tmpSigTens = np.array([[tmpSig[0],tmpSig[2],tmpSig[3]],\
                        [tmpSig[2],tmpSig[1],tmpSig[4]],\
                        [tmpSig[3],tmpSig[4],tmpSig[5]]])
                    eigs,trash = np.linalg.eig(tmpSigTens)
                    sigData[i,j] = min(eigs)
                elif crit=='sig_11':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[0]
                elif crit=='sig_22':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[1]
                elif crit=='sig_12':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[2]
                elif crit=='sig_13':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[3]
                elif crit=='sig_23':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[4]
                elif crit=='sig_33':
                    tmpSig = sigState[:,tmpInd]
                    sigData[i,j] = tmpSig[5]
                elif crit=='none':
                    sigData[i,j] = 0.
                
                    
        return sigData
    
    def printSummary(self,nodes=False):
        """A method for printing a summary of the CQUADX element.
        
        Prints out a tabulated form of the element ID, as well as the node ID's
        referenced by the element.
        
        :Args:
        
        - None
            
        :Returns:
        
        - `summary (str)`: Prints the tabulated EID, node IDs and material IDs
            associated with the CQUADX element.
            
        """
        print('CQUADX Summary:')
        headers = ('EID','NID 1','NID 2','NID 3','NID 4','MID')
        print(tabulate([[self.EID]+self.NIDs+[self.MID]],headers,tablefmt="fancy_grid"))
        if nodes:
            for node in self.nodes:
                node.printSummary()
    def clearXSectionMatricies(self):
        """Clears large matricies associated with cross-sectional analaysis.
        
        Intended primarily as a private method but left public, this method
        clears the matricies associated with cross-sectional analysis. This is
        mainly done as a way of saving memory.
        
        """
        self.Ae = None
        self.Ce = None
        self.Ee = None
        self.Le = None
        self.Me = None
        self.Re = None

class MaterialLib:
    """Creates a material library object.
    
    This material library holds the materials to be used for any type of
    analysis. Furthermore, it can be used to generate new material objects
    to be automatically stored within it. See the Material class for suported
    material types.
    
    :Attributes:
    
    - `matDict (dict)`: A dictionary which stores material objects as the
        values with the MIDs as the associated keys.
        
    :Methods:
    
    - `addMat`: Adds a material to the MaterialLib object dictionary.
    - `getMat`: Returns a material object provided an MID
    - `printSummary`: Prints a summary of all of the materials held within the
        matDict dictionary.
        
    """
    def __init__(self):
        """Initialize MaterialLib object.
        
        The initialization method is mainly used to initialize a dictionary
        which houses material objects.
        
        :Args:
        
        - None
            
        :Returns:
        
        - None
            
        """
        self.matDict = {}
    def addMat(self,MID, mat_name, mat_type, mat_constants,mat_t=0.,**kwargs):
        """Add a material to the MaterialLib object.
        
        This is the primary method of the class, used to create new material
        obects and then add them to the library for later use.
        
        :Args:
        
        - `MID (int)`: Material ID.
        - `name (str)`: Name of the material.
        - `matType (str)`: The type of the material. Supported material types
            are "iso", "trans_iso", and "ortho".
        - `mat_constants (1xX Array[Float])`: The requisite number of material
            constants required for any structural analysis. Note, this
            array includes the material density. For example, an isotropic
            material needs 2 elastic material constants, so the total
            length of mat_constants would be 3, 2 elastic constants and the
            density.
        - `mat_t (float)`: The thickness of 1-ply of the material
        - `th (1x3 Array[float])`: The angles about which the material can be
            rotated when it is initialized. In degrees.
        - `overwrite (bool)`: Input used in order to define whether the
            material being added can overwrite another material already
            held by the material library with the same MID.
            
        :Returns:
        
        - None
            
        """
        # Whether a material has the right to overwrite other materials
        overwrite = kwargs.pop('overwrite',False)
        # Optional argument for material direction rotation
        th = kwargs.pop('th', [0,0,0])
        # Check to see if there is a danger of overwriting a material
        if MID in self.matDict.keys() and not overwrite:
            raise StandardError('You may not overwrite a library material'+\
                ' entry without adding the optional argument overwrite=True')
        # Save material
        self.matDict[MID] = Material(MID, mat_name, mat_type, mat_constants,mat_t,th=th)
    def getMat(self,MID):
        """Method that returns a material from the material libary
        
        :Args:
        
        - `MID (int)`: The ID of the material which is desired
            
        :Returns:
        
        - `(obj): A material object associated with the key MID
            
        """
        if not MID in self.matDict.keys():
            raise KeyError('The MID provided is not linked with any materials'+
                'within the supplied material library.')
        return self.matDict[MID]
    def printSummary(self):
        """Prints summary of all Materials in MaterialLib
        
        A method used to print out tabulated summary of all of the materials
        held within the material library object.
        
        :Args:
        
        - None
            
        :Returns:
        
        - (str): A tabulated summary of the materials.
            
        """
        if len(self.matDict)==0:
            print('The material library is currently empty.\n')
        else:
            print('The materials are:')
            for mat in self.matDict:
                self.matDict[mat].printSummary()
        
        
class Ply:
    """Creates a CLT ply object.
    
    A class inspired by CLT, this class can be used to generate laminates
    to be used for CLT or cross-sectional analysis. It is likely that ply
    objects won't be created individually and then assembeled into a lamiante.
    More likely is that the plies will be generated within the laminate object.
    It should also be noted that it is assumed that the materials used are
    effectively at most transversely isotropic.
    
    :Attributes:
    
    - `E1 (float)`: Stiffness in the fiber direction.
    - `E2 (float)`: Stiffness transverse to the fiber direction.
    - `nu_12 (float)`: In plane poisson ratio.
    - `G_12 (float)`: In plane shear modulus.
    - `t (float)`: Thickness of the ply.
    - `Qbar (1x6 np.array[float])`: The terms in the rotated, reduced stiffness
        matrix. Ordering is as follows: [Q11,Q12,Q16,Q22,Q26,Q66]
    - `MID (int)`: An integer refrencing the material ID used for the
        constitutive relations.
    - `th (float)`: The angle about which the fibers are rotated in the plane
        in degrees.
        
    :Methods:
    
    - `genQ`: Given the in-plane stiffnesses used by the material of the ply,
        the method calculates the terms of ther reduced stiffness matrix.
    - `printSummary`: This prints out a summary of the object, including
        thickness, referenced MID and in plane angle orientation theta in
        degrees.
        
    """
    def __init__(self,Material,th):
        """Initializes the ply.
        
        This method initializes information about the ply such as in-plane
        stiffness repsonse.
        
        :Args:
        
        - `Material (obj)`: A material object, most likely coming from a
            material library.
        - `th (float)`: The angle about which the fibers are rotated in the
            plane in degrees.
            
        :Returns:
        
        - None
            
        """
        self.E1 = Material.E1
        self.E2 = Material.E2
        self.nu_12 = Material.nu_12
        self.G_12 = Material.G_12
        self.t = Material.t
        self.Q = self.genQ(self.E1,self.E2,self.nu_12,self.G_12)
        self.Qbar = self.rotRedStiffMat(self.Q,th)
        self.QbarMat = np.array([[self.Qbar[0],self.Qbar[1],self.Qbar[2]],\
                                 [self.Qbar[1],self.Qbar[3],self.Qbar[4]],\
                                 [self.Qbar[2],self.Qbar[4],self.Qbar[5]]])
        self.MID = Material.MID
        self.th = th
    def genQ(self,E1,E2,nu12,G12):
        """A method for calculating the reduced compliance of the ply.
        
        Intended primarily as a private method but left public, this method,
        for those unfarmiliar with CLT, calculates the terms in the reduced stiffness
        matrix given the in plane ply stiffnesses. It can be thus inferred that
        this requires the assumption of plane stres. This method is primarily
        used during the ply instantiation.
        
        :Args:
        
        - `E1 (float)`: The fiber direction stiffness.
        - `E2 (float)`: The stiffness transverse to the fibers.
        - `nu12 (float)`: The in-plane poisson ratio.
        - `G12 (float)`: The in-plane shear stiffness.
            
        :Returns:
        
        - `(1x4 np.array[float])`: The terms used in the reduced stiffness
            matrix. The ordering is: [Q11,Q12,Q22,Q66].
            
        """
        # Calculate the other in-plane poisson ratio.
        nu21 = nu12*E2/E1
        return [E1/(1-nu12*nu21),nu12*E2/(1-nu12*nu21),E2/(1-nu12*nu21),G12]
    def rotRedStiffMat(self,Q,th):
        """Calculate terms in the rotated, reduced stiffness matrix.
        
        Intended primarily as a private method but left public, this method,
        this method is used to rotate the plies reduced compliance matrix to
        the local laminate coordinate system.
        
        :Args:
        
        - `Q (1x4 np.array[float])`: The reduced compliance array containing
            [Q11,Q12,Q22,Q66]
        - `th(float)`: The angle the fibers are to be rotated in plane of the
            laminate.
            
        :Returns:
        
        - `(1x6 np.array[float])`: The reduced and rotated stiffness matrix terms
            for the ply. The ordering is: [Q11, Q12, Q16, Q22, Q26, Q66].
            
        """
        # Convert the angle to radians
        th = np.deg2rad(th)
        # Pre-calculate cosine of theta
        m = np.cos(th)
        # Pre-calculate sine of theta
        n = np.sin(th)
        # Compute the rotated, reduced stiffness matrix terms:
        Q11bar = Q[0]*m**4+2*(Q[1]+2*Q[3])*n**2*m**2+Q[2]*n**4
        Q12bar = (Q[0]+Q[2]-4*Q[3])*n**2*m**2+Q[1]*(n**4+m**4)
        Q16bar = (Q[0]-Q[1]-2*Q[3])*n*m**3+(Q[1]-Q[2]+2*Q[3])*n**3*m
        Q22bar = Q[0]*n**4+2*(Q[1]+2*Q[3])*n**2*m**2+Q[2]*m**4
        Q26bar = (Q[0]-Q[1]-2*Q[3])*n**3*m+(Q[1]-Q[2]+2*Q[3])*n*m**3
        Q66bar = (Q[0]+Q[2]-2*Q[1]-2*Q[3])*n**2*m**2+Q[3]*(n**4+m**4)
        return [Q11bar,Q12bar,Q16bar,Q22bar,Q26bar,Q66bar]
    def printSummary(self):
        """Prints a summary of the ply object.
        
        A method for printing a summary of the ply properties, such as
        the material ID, fiber orientation and ply thickness.
        
        :Args:
        
        - None
            
        :Returns:
        
        - `(str)`: Printed tabulated summary of the ply.
            
        """
        headers = ['MID','Theta, degrees','Thickness']
        print(tabulate(([[self.MID,self.th, self.t]]),headers))
        
class Laminate:
    """Creates a CLT laminate object.
    
    This class has two main uses. It can either be used for CLT analysis, or it
    can be used to build up a 2D mesh for a descretized cross-section.
    
    :Attributes:
    
    - `mesh (NxM np.array[int])`: This 2D array holds NIDs and is used
        to represent how nodes are organized in the 2D cross-section of
        the laminate.
    - `xmesh (NxM np.array[int])`: This 2D array holds the rigid x-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `ymesh (NxM np.array[int])`: This 2D array holds the rigid y-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `zmesh (NxM np.array[int])`: This 2D array holds the rigid z-coordinates
        of the nodes within the 2D descretization of the laminate on the
        local xsect CSYS.
    - `H (float)`: The total laminate thickness.
    - `rho_A (float)`: The laminate area density.
    - `plies (1xN array[obj])`: Contains an array of ply objects used to
        construct the laminate.
    - `t (1xN array[float])`: An array containing all of the ply thicknesses.
    - `ABD (6x6 np.array[float])`: The CLT 6x6 matrix relating in-plane strains
        and curvatures to in-plane force and moment resultants.
    - `abd (6x6 np.array[float])`: The CLT 6x6 matrix relating in-plane forces
        and moments resultants to in-plane strains and curvatures.
    - `z (1xN array[float])`: The z locations of laminate starting and ending
        points. This system always starts at -H/2 and goes to H/2
    - `equivMat (obj)`: This is orthotropic material object which exhibits
        similar in-plane stiffnesses.
    - `forceRes (1x6 np.array[float])`: The applied or resulting force and
        moment resultants generated during CLT analysis.
    - `globalStrain (1x6 np.array[float])`:  The applied or resulting strain
        and curvatures generated during CLT analysis.
        
        
    :Methods:
    
    - `printSummary`: This method prints out defining attributes of the
        laminate, such as the ABD matrix and layup schedule.
        
    """
    def __init__(self,n_i_tmp,m_i_tmp,matLib,**kwargs):
        """Initializes the Laminate object
        
        The way the laminate initialization works is you pass in two-three
        arrays and a material library. The first array contains information
        about how many plies you want to stack, the second array determines
        what material should be used for those plies, and the third array
        determines at what angle those plies lie. The class was developed this
        way as a means to fascilitate laminate optimization by quickly changing
        the number of plies at a given orientation and using a given material.
        
        :Args:
        
        - `n_i_tmp (1xN array[int])`: An array containing the number of plies
            using a material at a particular orientation such as:
            (theta=0,theta=45...)
        - `m_i_tmp (1xN array[int])`: An array containing the material to be
            used for the corresponding number of plies in the n_i_tmp array
        - `matLib (obj)`: The material library holding different material
            objects.
        - `sym (bool)`: Whether the laminate is symetric. (False by default)
        - `th (1xN array[float])`: An array containing the orientation at which
            the fibers are positioned within the laminate.
            
        :Returns:
        
        - None
            
        .. Note:: If you wanted to create a [0_2/45_2/90_2/-45_2]_s laminate of the
            same material, you could call laminate as:
            
            lam = Laminate([2,2,2,2],[1,1,1,1],matLib,sym=True)
            
            Or:
            
            lam = Laminate([2,2,2,2],[1,1,1,1],matLib,sym=True,th=[0,45,90,-45])
            
            Both of these statements are equivalent. If no theta array is
            provided and n_i_tmp is not equal to 4, then Laminate will default
            your fibers to all be running in the 0 degree orientation.
            
        """
        # Initialize attribute handles for latter X-Section meshing assignment
        self.mesh = None
        self.xmesh = None
        self.ymesh = None
        self.zmesh = None
        # Assign symetric laminate parameter
        sym = kwargs.pop('sym',False)
        # Verify that n_i_tmp and m_i_tmp are the same length
        if not len(n_i_tmp)==len(m_i_tmp):
            raise ValueError('n_i_tmp and m_i_tmp must be the same length.\n')
        # If no th provided, assign and n_i_tmp is a 4 length array, make
        # th=[0,45,90,-45].
        if len(n_i_tmp)==4:
            th = kwargs.pop('th',[0,45,90,-45])
        # Otherwise make th 0 for the length of n_i_tmp
        else:
            th = kwargs.pop('th',[0]*len(n_i_tmp))
        # If the laminate is symmetric, reflect n_i_tmp and m_i_tmp
        if sym:
            n_i_tmp = n_i_tmp+n_i_tmp[::-1]
            m_i_tmp = m_i_tmp+m_i_tmp[::-1]
            th = th+th[::-1]
        #Calculate the total laminate thickness and area density:
        H = 0.
        rho_A = 0.
        for i in range(0,len(th)):
            tmp_mat = matLib.matDict[m_i_tmp[i]]
            H += tmp_mat.t*n_i_tmp[i]
            rho_A += tmp_mat.t*n_i_tmp[i]*tmp_mat.rho
        # Assign the total laminate thickness H
        self.H = H
        # Assign the laminate area density
        self.rho_A = rho_A
        z = np.zeros(sum(n_i_tmp)+1)
        z[0] = -self.H/2.
        # Initialize ABD Matrix, thermal and moisture unit forces, and the area
        # density.
        ABD = np.zeros((6,6))
        #TODO: Add thermal and moisture support
        # NM_T = np.zeros((6,1))
        # NM_B = np.zeros((6,1))
        # Counter for ease of programming. Could go back and fix:
        c = 0
        # Initialize plies object array
        self.plies = []
        # Initialize thickness float array
        self.t = []
        # For all plies
        for i in range(0,len(th)):
            # Select the temporary material for the ith set of plies
            tmp_mat = matLib.matDict[m_i_tmp[i]]
            # For the number of times the ply material and orientation are
            # repeated
            for j in range(0,n_i_tmp[i]):
                # Create a new ply
                tmp_ply = Ply(tmp_mat,th[i])
                # Add the new ply to the array of plies held by the laminate
                self.plies+=[tmp_ply]
                # Update z-position array
                z[c+1] = z[c]+tmp_mat.t
                # Add terms to the ABD matrix for laminate reponse
                ABD[0:3,0:3] += tmp_ply.QbarMat*(z[c+1]-z[c])
                ABD[0:3,3:6] += (1./2.)*tmp_ply.QbarMat*(z[c+1]**2-z[c]**2)
                ABD[3:6,0:3] += (1./2.)*tmp_ply.QbarMat*(z[c+1]**2-z[c]**2)
                ABD[3:6,3:6] += (1./3.)*tmp_ply.QbarMat*(z[c+1]**3-z[c]**3)
                c += 1
                # Create array of all laminate thicknesses
                self.t += [tmp_mat.t]
        # Assign the ABD matrix to the object
        self.ABD = ABD
        # Assign the inverse of the ABD matrix to the object
        self.abd = np.linalg.inv(ABD)
        # Assign the coordinates for the laminate (demarking the interfaces
        # between plies within the laminate) to the object
        self.z = z
        # Generate equivalent in-plane engineering properties:
        Ex = (ABD[0,0]*ABD[1,1]-ABD[0,1]**2)/(ABD[1,1]*H)
        Ey = (ABD[0,0]*ABD[1,1]-ABD[0,1]**2)/(ABD[0,0]*H)
        G_xy = ABD[2,2]/H
        nu_xy = ABD[0,1]/ABD[1,1]
        # nuyx = ABD[0,1]/ABD[0,0]
        mat_constants = [Ex, Ey, nu_xy, 0., G_xy, rho_A]
        # Create an equivalent material object for the laminate
        self.equivMat = Material(101, 'Equiv Lam Mat', 'trans_iso', mat_constants,H)
        # Initialize Miscelanoes Parameters:
        self.forceRes = np.zeros(6)
        self.globalStrain = np.zeros(6)
    def printSummary(self,**kwargs):
        """Prints a summary of information about the laminate.
        
        This method can print both the ABD matrix and ply information schedule
        of the laminate.
        
        :Args:
        
        - `ABD (bool)`: This optional argument asks whether the ABD matrix
            should be printed.
        - `decimals (int)`: Should the ABD matrix be printed, python should
            print up to this many digits after the decimal point.
        - `plies (bool)`: This optional argument asks whether the ply schedule
            for the laminate should be printed.
            
        :Returns:
        
        - None
            
        """
        ABD = kwargs.pop('ABD',True)
        decimals = kwargs.pop('decimals',4)
        plies = kwargs.pop('plies',True)
        if ABD:
            print('ABD Matrix:')
            print(tabulate(np.around(self.ABD,decimals=decimals),tablefmt="fancy_grid"))
        if plies:
            for ply in self.plies:
                ply.printSummary()
    def plotLaminate(self,**kwargs):
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
        
class Mesher:
    """Meshes cross-section objects
    
    This class is used to descritize cross-sections provided laminate objects.
    Currently only two cross-sectional shapes are supported. The first is a
    box beam using an airfoil outer mold line, and the second is a hollow tube
    using as many laminates as desired. One of the main results is the
    population of the nodeDict and elemDict attributes for the cross-section.
    
    :Attributes:
    
    - None
        
    :Methods:
    
    - `boxBeam`: Taking several inputs including 4 laminate objects and meshes
        a 2D box beam cross-section.
    - `laminate`: Meshes the cross-section of a single laminate.
    - `cylindricalTube`: Taking several inputs including n laminate objects and
        meshes a 2D cylindrical tube cross-section.
    - `rectBoxBeam`: Meshes a rectangular cross-section, but it is more
        restrictive than boxBeam method. In this method, each of the four
        laminates must have the same number of plies, each of which are the
        same thickness.
        
    """
    def boxBeam(self,xsect,meshSize,x0,xf,matlib):
        """Meshes a box beam cross-section.
        
        This meshing routine takes several parameters including a cross-section
        object `xsect`. This cross-section object should also contain the
        laminate objects used to construct it. There are no restrictions place
        on these laminates. Furthermore the outer mold line of this cross-
        section can take the form of any NACA 4-series airfoil. Finally, the
        convention is that for the four laminates that make up the box-beam,
        the the first ply in the laminate (which in CLT corresponds to the last
        ply in the stack) is located on the outside of the box beam. This
        convention can be seen below:
        
        .. image:: images/boxBeamGeom.png
            :align: center
        
        :Args:
        
        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.
            
        :Returns:
        
        - None
            
        """
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # The chord length of the airfoil profile
        c = Airfoil.c
        # Initialize the z location of the cross-section
        zc = 0
        # Initialize the Euler angler rotation about the local xsect z-axis for
        # any the given laminate. Note that individual elements might
        # experience further z-axis orientation if there is curvature in in the
        # OML of the cross-section.
        thz = [0,90,180,270]
        
        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==4:
            raise ValueError('The box beam cross-section was selected, but 4 '\
                'laminates were not provided')
        # Determine the number of plies per each laminate
        nlam1 = len(laminates[0].plies)
        nlam2 = len(laminates[1].plies)
        nlam3 = len(laminates[2].plies)
        nlam4 = len(laminates[3].plies)
        # Define boundary curves:
        # Note, the following curves represent the x-coordinate mesh
        # seeding along key regions, such as the connection region
        # between laminate 1 and 2
        x2 = np.zeros(len(laminates[1].plies))
        x4 = np.zeros(len(laminates[3].plies))
        x3 = np.linspace(x0+laminates[1].H/c,xf-laminates[3].H/c,int(((xf-laminates[3].H/c)\
            -(x0+laminates[1].H/c))/(meshSize*min(laminates[0].t)/c)))[1:]
        x5 = np.linspace(x0+laminates[1].H/c,xf-laminates[3].H/c,int(((xf-laminates[3].H/c)\
            -(x0+laminates[1].H/c))/(meshSize*min(laminates[2].t)/c)))[1:]
        # Populates the x-coordinates of the mesh seeding in curves x2 and
        # x4, which are the joint regions between the 4 laminates.
        x2 = x0+(laminates[1].z+laminates[1].H/2)/c
        x4 = xf-(laminates[3].z[::-1]+laminates[3].H/2)/c

        x1top = np.hstack((x2,x3,x4[1:]))
        x3bot = np.hstack((x2,x5,x4[1:]))
        
        # GENERATE LAMINATE 1 AND 3 MESHES
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 1 and 3). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        lam1Mesh = np.zeros((1+nlam1,len(x1top)),dtype=int)
        lam1xMesh = np.zeros((1+nlam1,len(x1top)))
        lam1yMesh = np.zeros((1+nlam1,len(x1top)))
        lam3Mesh = np.zeros((1+nlam3,len(x3bot)),dtype=int)
        lam3xMesh = np.zeros((1+nlam3,len(x3bot)))
        lam3yMesh = np.zeros((1+nlam3,len(x3bot)))
        #Generate the xy points of the top airfoil curve
        xu,yu,trash1,trash2 = Airfoil.points(x1top)
        #Generate the xy points of the bottom airfoil curve
        trash1,trash2,xl,yl = Airfoil.points(x3bot)
        #Generate the node objects for laminate 1
        ttmp = [0]+(laminates[0].z+laminates[0].H/2)
        for i in range(0,nlam1+1):
            for j in range(0,len(x1top)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam1Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xu[j],yu[j]-ttmp[i],zc]))
                lam1xMesh[i,j] = xu[j]
                lam1yMesh[i,j] = yu[j]-ttmp[i]
        #Generate  the node objects for laminate 3
        ttmp = [0]+laminates[2].z+laminates[2].H/2
        for i in range(0,nlam3+1):
            for j in range(0,len(x3bot)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam3Mesh[-1-i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xl[j],yl[j]+ttmp[i],zc]))
                lam3xMesh[-1-i,j] = xl[j]
                lam3yMesh[-1-i,j] = yl[j]+ttmp[i]
        #GENERATE LAMINATE 2 AND 4 MESHES
        #Define the mesh seeding for laminate 2
        meshLen2 = int(((yu[0]-laminates[0].H)-(yl[0]+laminates[2].H))/(meshSize*min(laminates[1].t)))
        #Define the mesh seeding for laminate 4
        meshLen4 = int(((yu[-1]-laminates[0].H)-(yl[-1]+laminates[2].H))/(meshSize*min(laminates[3].t)))
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 2 and 4). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        lam2Mesh = np.zeros((meshLen2,nlam2+1),dtype=int)
        lam2xMesh = np.zeros((meshLen2,nlam2+1))
        lam2yMesh = np.zeros((meshLen2,nlam2+1))
        lam4Mesh = np.zeros((meshLen4,nlam4+1),dtype=int)
        lam4xMesh = np.zeros((meshLen4,nlam4+1))
        lam4yMesh = np.zeros((meshLen4,nlam4+1))
        #Add connectivity nodes for lamiante 2
        lam2Mesh[0,:] = lam1Mesh[-1,0:nlam2+1]
        lam2xMesh[0,:] = lam1xMesh[-1,0:nlam2+1]
        lam2yMesh[0,:] = lam1yMesh[-1,0:nlam2+1]
        lam2Mesh[-1,:] = lam3Mesh[0,0:nlam2+1]
        lam2xMesh[-1,:] = lam3xMesh[0,0:nlam2+1]
        lam2yMesh[-1,:] = lam3yMesh[0,0:nlam2+1]
        #Generate the node objects for laminate 2
        for i in range(0,nlam2+1):
            lam2xMesh[:,i] = np.linspace(lam2xMesh[0,i],lam2xMesh[-1,i],meshLen2).T
            lam2yMesh[:,i] = np.linspace(lam2yMesh[0,i],lam2yMesh[-1,i],meshLen2).T
            for j in range(1,np.size(lam2xMesh,axis=0)-1):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam2Mesh[j,i] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam2xMesh[j,i],lam2yMesh[j,i],zc]))
        #Add connectivity nodes for lamiante 4
        lam4Mesh[0,:] = lam1Mesh[-1,-(nlam2+1):]
        lam4xMesh[0,:] = lam1xMesh[-1,-(nlam2+1):]
        lam4yMesh[0,:] = lam1yMesh[-1,-(nlam2+1):]
        lam4Mesh[-1,:] = lam3Mesh[0,-(nlam2+1):]
        lam4xMesh[-1,:] = lam3xMesh[0,-(nlam2+1):]
        lam4yMesh[-1,:] = lam3yMesh[0,-(nlam2+1):]
        #Generate the node objects for laminate 4
        for i in range(0,nlam4+1):
            lam4xMesh[:,i] = np.linspace(lam4xMesh[0,i],lam4xMesh[-1,i],meshLen4).T
            lam4yMesh[:,i] = np.linspace(lam4yMesh[0,i],lam4yMesh[-1,i],meshLen4).T
            for j in range(1,np.size(lam4Mesh,axis=0)-1):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam4Mesh[j,i] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam4xMesh[j,i],lam4yMesh[j,i],zc]))
        # Save meshes:
        xsect.laminates[0].mesh = lam1Mesh
        xsect.laminates[0].xmesh = lam1xMesh
        xsect.laminates[0].ymesh = lam1yMesh
        xsect.laminates[0].zmesh = np.zeros((1+nlam1,len(x1top)))

        xsect.laminates[1].mesh = lam2Mesh
        xsect.laminates[1].xmesh = lam2xMesh
        xsect.laminates[1].ymesh = lam2yMesh
        xsect.laminates[1].zmesh = np.zeros((meshLen2,nlam2+1))
        
        xsect.laminates[2].mesh = lam3Mesh
        xsect.laminates[2].xmesh = lam3xMesh
        xsect.laminates[2].ymesh = lam3yMesh
        xsect.laminates[2].zmesh = np.zeros((1+nlam3,len(x3bot)))
        
        xsect.laminates[3].mesh = lam4Mesh
        xsect.laminates[3].xmesh = lam4xMesh
        xsect.laminates[3].ymesh = lam4yMesh
        xsect.laminates[3].zmesh = np.zeros((meshLen4,nlam4+1))
        
        xsect.nodeDict = nodeDict
        xsect.xdim = max([np.max(lam1xMesh),np.max(lam2xMesh),np.max(lam3xMesh),np.max(lam4xMesh)])\
            -max([np.min(lam1xMesh),np.min(lam2xMesh),np.min(lam3xMesh),np.min(lam4xMesh)])
        xsect.ydim = max([np.max(lam1yMesh),np.max(lam2yMesh),np.max(lam3yMesh),np.max(lam4yMesh)])\
            -max([np.min(lam1yMesh),np.min(lam2yMesh),np.min(lam3yMesh),np.min(lam4yMesh)])
        
        for k in range(0,len(xsect.laminates)):
            ylen = np.size(xsect.laminates[k].mesh,axis=0)-1
            xlen = np.size(xsect.laminates[k].mesh,axis=1)-1
            # Ovearhead for later plotting of the cross-section. Will allow
            # for discontinuities in the contour should it arise (ie in
            # stress or strain contours).
            xsect.laminates[k].plotx = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].ploty = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotz = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotc = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].EIDmesh = np.zeros((ylen,xlen),dtype=int)
            for i in range(0,ylen):
                for j in range(0,xlen):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [xsect.laminates[k].mesh[i+1,j],xsect.laminates[k].mesh[i+1,j+1],\
                        xsect.laminates[k].mesh[i,j+1],xsect.laminates[k].mesh[i,j]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    # If the laminate is horizontal (i.e. divisible by 2)
                    if k % 2==0:
                        # Section determines how curvature in the beam causes
                        # slight variations in fiber rotation.
                        deltax1 = xsect.laminates[k].xmesh[i,j+1]-xsect.laminates[k].xmesh[i,j]
                        deltay1 = xsect.laminates[k].ymesh[i,j+1]-xsect.laminates[k].ymesh[i,j]
                        deltax2 = xsect.laminates[k].xmesh[i+1,j+1]-xsect.laminates[k].xmesh[i+1,j]
                        deltay2 = xsect.laminates[k].ymesh[i+1,j+1]-xsect.laminates[k].ymesh[i+1,j]
                        thz_loc = np.rad2deg(np.mean([np.arctan(deltay1/deltax1), np.arctan(deltay2/deltax2)]))
                        if k==0:
                            MID = xsect.laminates[k].plies[ylen-i-1].MID
                            th = [0,xsect.laminates[k].plies[ylen-i-1].th,thz[k]+thz_loc]
                        else:
                            MID = xsect.laminates[k].plies[i].MID
                            th = [0,xsect.laminates[k].plies[i].th,thz[k]+thz_loc]
                        
                        #if newEID in [0,1692,1135,1134,2830,2831]:
                        #    print(th)
                    # Else if it is vertical:
                    else:
                        if k==1:
                            MID = xsect.laminates[k].plies[xlen-j-1].MID
                            th = [0,xsect.laminates[k].plies[xlen-j-1].th,thz[k]]
                        else:
                            MID = xsect.laminates[k].plies[j].MID
                            th = [0,xsect.laminates[k].plies[j].th,thz[k]]
                        #MID = xsect.laminates[k].plies[j].MID
                        
                        #if newEID in [0,1692,1135,1134,2830,2831]:
                        #    print(th)
                    elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib,th=th)
                    xsect.laminates[k].EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def laminate(self,xsect,meshSize,x0,xf,matlib):
        """Meshes laminate cross-section.
        
        This method meshes a simple laminate cross-section. It is assumed that
        the unit normal vector of the laminate points in the y-direction. This
        method only requires one laminate, which can take any shape. The cross-
        section geometry can be seen below:
        
        .. image:: images/laminateGeom.png
            :align: center
        
        :Args:
        
        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.
            
        :Returns:
        
        - None
            
        """
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # The chord length of the airfoil profile
        c = Airfoil.c
        
        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==1:
            raise ValueError('The laminate cross-section was selected, but 1 '\
                'laminate was not provided')
        # Determine the number of plies per each laminate
        laminate = laminates[0]
        # get the y coordinates of the lamiante
        ycoords = laminate.z[::-1]
        xcoords = np.linspace(x0*c,xf*c,int(c/(min(laminate.t)*meshSize))+1)
        # Create 2D meshes
        xmesh,ymesh = np.meshgrid(xcoords,ycoords)
        # Create z-mesh
        zmesh = np.zeros((len(ycoords),len(xcoords)))
        
        Mesh = np.zeros((len(ycoords),len(xcoords)),dtype=int)
        
        for i in range(0,len(ycoords)):
            for j in range(0,len(xcoords)):
                newNID = max(nodeDict.keys())+1
                Mesh[i,j] = newNID
                nodeDict[newNID] = Node(newNID,[xmesh[i,j],ymesh[i,j],zmesh[i,j]])
        
        # Save meshes:
        laminate.mesh = Mesh
        laminate.xmesh = xmesh
        laminate.ymesh = ymesh
        laminate.zmesh = zmesh
        
        xsect.nodeDict = nodeDict
        
        #Create Elements
        ylen = len(ycoords)-1
        xlen = len(xcoords)-1
        # Ovearhead for later plotting of the cross-section. Will allow
        # for discontinuities in the contour should it arise (ie in
        # stress or strain contours).
        laminate.plotx = np.zeros((ylen*2,xlen*2))
        laminate.ploty = np.zeros((ylen*2,xlen*2))
        laminate.plotz = np.zeros((ylen*2,xlen*2))
        laminate.plotc = np.zeros((ylen*2,xlen*2))
        laminate.EIDmesh = np.zeros((ylen,xlen),dtype=int)
        for i in range(0,ylen):
            for j in range(0,xlen):
                newEID = int(max(elemDict.keys())+1)
                NIDs = [laminate.mesh[i+1,j],laminate.mesh[i+1,j+1],\
                    laminate.mesh[i,j+1],laminate.mesh[i,j]]
                nodes = [xsect.nodeDict[NID] for NID in NIDs]
                MID = laminate.plies[ylen-1-i].MID
                th = [0,laminate.plies[i].th,0.]
                elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib,th=th)
                laminate.EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def cylindricalTube(self,xsect,r,meshSize,x0,xf,matlib,**kwargs):
        # Initialize the node dictionary, containing all local node objects
        # used by the cross-section
        nodeDict = {-1:None}
        # Initialize the node dictionary, containing all local element objects
        # used by the cross-section
        elemDict = {-1:None}
        # Initialize the X-Section z-coordinate
        zc = kwargs.pop('zc',0)
        # Initialize the laminates
        laminates = xsect.laminates
        # Initialize the number of plies per laminate (must be equal for all)
        nplies = len(laminates[0].plies)
        # Initialize the thickness vectors of plies per laminate (must be equal for all)
        ts = laminates[0].t
        # Determine the dtheta required for the cross-section
        minT = 1e9
        for lam in laminates:
            lamMin = min(lam.t)
            if lamMin<minT:
                minT = lamMin
            # Check the total number of laminates
            if not len(lam.plies)==nplies:
                raise ValueError('Note, for now all laminates must have the'\
                    'same number of plies.')
            # Check that the thicknesses all match
            if not np.array_equal(ts,lam.t):
                raise ValueError('Note, for now all laminates must have the'\
                    'Sane thickness distribution through the thickness of the'\
                    'laminate in order to preserve mesh compatability between'\
                    'laminates.')
        dth = meshSize*minT/r
        thz = []
        for i in range(0,len(laminates)):
            thz = np.append(thz,np.linspace(i*2*np.pi/len(laminates),\
                (i+1)*2*np.pi/len(laminates)),num=int(2*np.pi/(dth*len(laminates))))
        thz = np.unique(thz[0:-1])
        rvec = r+laminates[0].z+laminates[0].H/2
        rmat,thmat = np.meshgrid(rvec,thz)
        mesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)),dtype=int)
        xmesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        ymesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        zmesh = np.zeros((np.size(rmat,axis=0),np.size(rmat,axis=1)))
        for i in range(0,np.size(rmat,axis=0)):
            for j in range(0,np.size(rmat,axis=1)):
                # Determine temp xy coordinates of the point
                xtmp = rmat[i,j]*np.cos(thmat[i,j])
                ytmp = rmat[i,j]*np.sin(thmat[i,j])
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xtmp,ytmp,zc]))
                xmesh[i,j] = xtmp
                ymesh[i,j] = ytmp
        # Assign parts of the total mesh to each laminate
        bound = np.linspace(0,1,num=len(laminates)+1)
        for i in range(0,len(laminates)):
            laminates[i].mesh = mesh[(thmat<=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].xmesh = xmesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].ymesh = ymesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].zmesh = zmesh[(thmat>=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].thmesh = thmat[(thmat<=bound[i]) & (thmat<=bound[i+1])]
            laminates[i].EIDmesh = np.zeros((np.size(laminates[i].mesh,axis=0)\
                ,np.size(laminates[i].mesh,axis=1)),dtype=int)
        for lam in laminates:
            for i in range(0,np.size(lam.mesh,axis=0)-1):
                for j in range(0,np.size(lam.mesh,axis=1)-1):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [lam.mesh[i+1,j+1],lam.mesh[i+1,j],\
                        lam.mesh[i,j],lam.mesh[i,j+1]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    th = [0,lam.plies[i].th,lam.thmesh[i,j]]
                    MID = xsect.lam.plies[i].MID
                    elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib,th=th)
                    xsect.lam.EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def rectBoxBeam(self,xsect,meshSize,x0,xf,matlib):
        """Meshes a box beam cross-section.
        
        This method meshes a similar cross-section as the boxBeam method. The
        geometry of this cross-section can be seen below. The interfaces
        between the laminates is different, and more restrictive. In this case
        all of the laminates must have the same number of plies, which must
        also all be the same thickness.
        
        .. image:: images/rectBoxGeom.png
            :align: center
        
        :Args:
        
        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.
            
        :Returns:
        
        - None
            
        """
        print('Rectangular Box Meshing Commencing')
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # The chord length of the airfoil profile
        c = Airfoil.c
        # Initialize the z location of the cross-section
        zc = 0
        # Initialize the Euler angler rotation about the local xsect z-axis for
        # any the given laminate. Note that individual elements might
        # experience further z-axis orientation if there is curvature in in the
        # OML of the cross-section.
        thz = [0,90,180,270]
        
        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==4:
            raise ValueError('The box beam cross-section was selected, but 4 '\
                'laminates were not provided')
        # Determine the number of plies per each laminate
        nlam1 = len(laminates[0].plies)
        nlam2 = len(laminates[1].plies)
        nlam3 = len(laminates[2].plies)
        nlam4 = len(laminates[3].plies)
        # Define boundary curves:
        # Note, the following curves represent the x-coordinate mesh
        # seeding along key regions, such as the connection region
        # between laminate 1 and 2
        
        # Populates the x-coordinates of the mesh seeding in curves x2 and
        # x4, which are the joint regions between the 4 laminates.
        
        
        # Calculate important x points:
        x0 = x0*c
        x1 = x0+laminates[1].H
        xf = xf*c
        x2 = xf-laminates[3].H
        
        # Calculate important y points:
        y0 = -c/2
        y1 = y0+laminates[2].H
        yf = c/2
        y2 = yf-laminates[0].H
        
        # Determine the mesh seeding to maintain minimum AR
        lam13xSeeding = np.ceil((xf-x0)/(meshSize*min(laminates[0].t)))
        lam24ySeeding = np.ceil((yf-y0)/(meshSize*min(laminates[0].t)))
        
        # Define Finite Element Modeling Functions
        def x(eta,xi,xs):
            return .25*(xs[0]*(1.-xi)*(1.-eta)+xs[1]*(1.+xi)*(1.-eta)+\
                    xs[2]*(1.+xi)*(1.+eta)+xs[3]*(1.-xi)*(1.+eta))
        def y(eta,xi,ys):
            return .25*(ys[0]*(1.-xi)*(1.-eta)+ys[1]*(1.+xi)*(1.-eta)+\
                    ys[2]*(1.+xi)*(1.+eta)+ys[3]*(1.-xi)*(1.+eta))
        
        # Generate Grids in superelement space
        xis13 = np.linspace(-1,1,lam13xSeeding+1)
        etas13 = np.linspace(1,-1,nlam1+1)
        lam1Mesh = np.zeros((1+nlam1,len(xis13)),dtype=int)
        lam3Mesh = np.zeros((1+nlam3,len(xis13)),dtype=int)
        xis13, etas13 = np.meshgrid(xis13,etas13)
        lam1xMesh = x(etas13,xis13,[x1,x2,xf,x0])
        lam1yMesh = y(etas13,xis13,[y2,y2,yf,yf])
        lam3xMesh = x(etas13,xis13,[x0,xf,x2,x1])
        lam3yMesh = y(etas13,xis13,[y0,y0,y1,y1])
        
        # GENERATE LAMINATE 1 AND 3 MESHES
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 1 and 3). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        
        for i in range(0,np.size(lam1xMesh,axis=0)):
            for j in range(0,np.size(lam1xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam1Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam1xMesh[i,j],lam1yMesh[i,j],zc]))
        #Generate  the node objects for laminate 3
        #ttmp = [0]+laminates[2].z+laminates[2].H/2
        for i in range(0,np.size(lam3xMesh,axis=0)):
            for j in range(0,np.size(lam3xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam3Mesh[-1-i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam3xMesh[-1-i,j],lam3yMesh[-1-i,j],zc]))
        #GENERATE LAMINATE 2 AND 4 MESHES
        #Define the mesh seeding for laminate 2
        #meshLen2 = int(((yu[0]-laminates[0].H)-(yl[0]+laminates[2].H))/(meshSize*min(laminates[1].t)))
        #Define the mesh seeding for laminate 4
        #meshLen4 = int(((yu[-1]-laminates[0].H)-(yl[-1]+laminates[2].H))/(meshSize*min(laminates[3].t)))
        # Create 3 empty numpy arrays for each laminate (we will start with
        # lamiantes 2 and 4). The first is holds node ID's, the second and
        # third hold the corresponding x and y coordinates of the node
        
        xis24 = np.linspace(-1,1,nlam2+1)
        etas24 = np.linspace(1,-1,lam24ySeeding+1)
        lam2Mesh = np.zeros((len(etas24),1+nlam2),dtype=int)
        lam4Mesh = np.zeros((len(etas24),1+nlam4),dtype=int)
        xis24, etas24 = np.meshgrid(xis24,etas24)
        lam2xMesh = x(etas24,xis24,[x0,x1,x1,x0])
        lam2yMesh = y(etas24,xis24,[y0,y1,y2,yf])
        lam4xMesh = x(etas24,xis24,[x2,xf,xf,x2])
        lam4yMesh = y(etas24,xis24,[y1,y0,yf,y2])
        
        #Add connectivity nodes for lamiante 2
        lam2Mesh[0,:] = lam1Mesh[:,0]
        lam2xMesh[0,:] = lam1xMesh[:,0]
        lam2yMesh[0,:] = lam1yMesh[:,0]
        lam2Mesh[-1,:] = lam3Mesh[::-1,0]
        lam2xMesh[-1,:] = lam3xMesh[::-1,0]
        lam2yMesh[-1,:] = lam3yMesh[::-1,0]
        #Add connectivity nodes for lamiante 4
        lam4Mesh[0,:] = lam1Mesh[::-1,-1]
        lam4xMesh[0,:] = lam1xMesh[::-1,-1]
        lam4yMesh[0,:] = lam1yMesh[::-1,-1]
        lam4Mesh[-1,:] = lam3Mesh[:,-1]
        lam4xMesh[-1,:] = lam3xMesh[:,-1]
        lam4yMesh[-1,:] = lam3yMesh[:,-1]
        #Generate the node objects for laminate 2
        for i in range(1,np.size(lam2xMesh,axis=0)-1):
            for j in range(0,np.size(lam2xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam2Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam2xMesh[i,j],lam2yMesh[i,j],zc]))
        #Generate the node objects for laminate 4
        for i in range(1,np.size(lam2xMesh,axis=0)-1):
            for j in range(0,np.size(lam2xMesh,axis=1)):
                #Create node/populate mesh array
                newNID = int(max(nodeDict.keys())+1)
                lam4Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([lam4xMesh[i,j],lam4yMesh[i,j],zc]))
        # Save meshes:
        xsect.laminates[0].mesh = lam1Mesh
        xsect.laminates[0].xmesh = lam1xMesh
        xsect.laminates[0].ymesh = lam1yMesh
        xsect.laminates[0].zmesh = np.zeros((np.size(lam1Mesh,axis=0),np.size(lam1Mesh,axis=1)))

        xsect.laminates[1].mesh = lam2Mesh
        xsect.laminates[1].xmesh = lam2xMesh
        xsect.laminates[1].ymesh = lam2yMesh
        xsect.laminates[1].zmesh = np.zeros((np.size(lam2Mesh,axis=0),np.size(lam2Mesh,axis=1)))
        
        xsect.laminates[2].mesh = lam3Mesh
        xsect.laminates[2].xmesh = lam3xMesh
        xsect.laminates[2].ymesh = lam3yMesh
        xsect.laminates[2].zmesh = np.zeros((np.size(lam3Mesh,axis=0),np.size(lam3Mesh,axis=1)))
        
        xsect.laminates[3].mesh = lam4Mesh
        xsect.laminates[3].xmesh = lam4xMesh
        xsect.laminates[3].ymesh = lam4yMesh
        xsect.laminates[3].zmesh = np.zeros((np.size(lam4Mesh,axis=0),np.size(lam4Mesh,axis=1)))
        
        xsect.nodeDict = nodeDict
        
        for k in range(0,len(xsect.laminates)):
            ylen = np.size(xsect.laminates[k].mesh,axis=0)-1
            xlen = np.size(xsect.laminates[k].mesh,axis=1)-1
            # Ovearhead for later plotting of the cross-section. Will allow
            # for discontinuities in the contour should it arise (ie in
            # stress or strain contours).
            xsect.laminates[k].plotx = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].ploty = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotz = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].plotc = np.zeros((ylen*2,xlen*2))
            xsect.laminates[k].EIDmesh = np.zeros((ylen,xlen),dtype=int)
            for i in range(0,ylen):
                for j in range(0,xlen):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [xsect.laminates[k].mesh[i+1,j],xsect.laminates[k].mesh[i+1,j+1],\
                        xsect.laminates[k].mesh[i,j+1],xsect.laminates[k].mesh[i,j]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    if k==0:
                        MID = xsect.laminates[k].plies[-i-1].MID
                        th = [0,xsect.laminates[k].plies[-i-1].th,thz[k]]
                    elif k==1:
                        MID = xsect.laminates[k].plies[-j-1].MID
                        th = [0,xsect.laminates[k].plies[-j-1].th,thz[k]]
                    elif k==2:
                        MID = xsect.laminates[k].plies[i].MID
                        th = [0,xsect.laminates[k].plies[i].th,thz[k]]
                    else:
                        MID = xsect.laminates[k].plies[j].MID
                        th = [0,xsect.laminates[k].plies[j].th,thz[k]]
                    elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib,th=th)
                    xsect.laminates[k].EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def solidBox(self,xsect, elemX, elemY, L1, L2, MID, matlib, elemOrder):
        """Meshes a box beam cross-section.
        
        This method meshes a similar cross-section as the boxBeam method. The
        geometry of this cross-section can be seen below. The interfaces
        between the laminates is different, and more restrictive. In this case
        all of the laminates must have the same number of plies, which must
        also all be the same thickness.
        
        .. image:: images/rectBoxGeom.png
            :align: center
        
        :Args:
        
        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.
            
        :Returns:
        
        - None
            
        """
        print('Box Meshing Commencing')
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        # Initialize the z location of the cross-section
        zc = 0
        if elemOrder==1:
            nnx = elemX+1
            nny = elemY+1
        else:
            nnx = 2*elemX+1
            nny = 2*elemY+1
        # Create Mesh
        xvec = np.linspace(-L1/2,L1/2,nnx)
        yvec = np.linspace(-L2/2,L2/2,nny)[::-1]
        # NID Mesh
        Mesh = np.zeros((nny,nnx),dtype=int)
        EIDmesh = np.zeros((elemY,elemX),dtype=int)
        xmesh,ymesh = np.meshgrid(xvec,yvec)
        for i in range(0,nny):
            for j in range(0,nnx):
                newNID = int(max(nodeDict.keys())+1)
                Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xmesh[i,j],ymesh[i,j],zc]))
                
        xsect.nodeDict = nodeDict
        
        if elemOrder==1:
            for i in range(0,elemY):
                for j in range(0,elemX):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [Mesh[i+1,j],Mesh[i+1,j+1],Mesh[i,j+1],Mesh[i,j]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib)
                    EIDmesh[i,j] = newEID
        else:
            for i in range(0,elemY):
                for j in range(0,elemX):
                    newEID = int(max(elemDict.keys())+1)
                    NIDs = [Mesh[2*i+2,2*j],Mesh[2*i+2,2*j+1],Mesh[2*i+2,2*j+2],\
                    Mesh[2*i+1,2*j],Mesh[2*i+1,2*j+1],Mesh[2*i+1,2*j+2],\
                    Mesh[2*i,2*j],Mesh[2*i,2*j+1],Mesh[2*i,2*j+2]]
                    nodes = [xsect.nodeDict[NID] for NID in NIDs]
                    elemDict[newEID] = CQUADX9(newEID,nodes,MID,matlib)
                    EIDmesh[i,j] = newEID
        
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
        
    def rectangleHole(self,xsect, nelem, a, b, r, MID, matlib):
        """Meshes a box beam cross-section.
        
        This method meshes a similar cross-section as the boxBeam method. The
        geometry of this cross-section can be seen below. The interfaces
        between the laminates is different, and more restrictive. In this case
        all of the laminates must have the same number of plies, which must
        also all be the same thickness.
        
        .. image:: images/rectBoxGeom.png
            :align: center
        
        :Args:
        
        - `xsect (obj)`: The cross-section object to be meshed.
        - `meshSize (int)`: The maximum aspect ratio an element can have
        - `x0 (float)`: The non-dimensional starting point of the cross-section
            on the airfoil.
        - `xf (float)`: The non-dimesnional ending point of the cross-section
            on the airfoil.
        - `matlib (obj)`: The material library object used to create CQUADX
            elements.
            
        :Returns:
        
        - None
            
        """
        print('Box Meshing Commencing')
        # INITIALIZE INPUTS
        # Initialize the node dictionary containing all nodes objects used by
        # the cross-section
        nodeDict = {-1:None}
        # Initialize the element dictionary containing all element objects used
        # by the cross-section
        elemDict = {-1:None}
        nelem=nelem*8+1
        laminate = xsect.laminates[0]
        # Initialize the z location of the cross-section
        xs = [a/2.,a/2.,0.,-a/2.,-a/2.,-a/2.,0.,a/2.,a/2.]
        ys = [0.,b/2.,b/2.,b/2.,0.,-b/2.,-b/2.,-b/2.,0.]
        
        xsvec = np.array([])
        ysvec = np.array([])
        
        for i in range(0,len(xs)-1):
            xsvec = np.append(xsvec,np.linspace(xs[i],xs[i+1],nelem/8.+1)[:-1])
            ysvec = np.append(ysvec,np.linspace(ys[i],ys[i+1],nelem/8.+1)[:-1])
        
        xc = r*np.cos(np.linspace(0,2*np.pi,nelem))[:-1]
        yc = r*np.sin(np.linspace(0,2*np.pi,nelem))[:-1]
        
        if not len(xc)==len(xsvec):
            raise ValueError('Circle and square vectors dont match length.')
        
        xmesh = np.zeros((int(nelem/8-1),len(xc)))
        ymesh = np.zeros((int(nelem/8-1),len(xc)))
        zmesh = np.zeros((int(nelem/8-1),len(xc)))
        Mesh = np.zeros((int(nelem/8-1),len(xc)),dtype=int)
        
        for i in range(0,len(xc)):
            xmesh[:,i]=np.linspace(xc[i],xsvec[i],nelem/8-1)
            ymesh[:,i]=np.linspace(yc[i],ysvec[i],nelem/8-1)
        
        for i in range(0,np.size(xmesh,axis=0)):
            for j in range(0,np.size(xmesh,axis=1)):
                newNID = int(max(nodeDict.keys())+1)
                Mesh[i,j] = newNID
                #Add node to NID Dictionary
                nodeDict[newNID] = Node(newNID,np.array([xmesh[i,j],ymesh[i,j],zmesh[i,j]]))
        
        xmesh = np.hstack((xmesh,np.array([xmesh[:,0]]).T))
        ymesh = np.hstack((ymesh,np.array([ymesh[:,0]]).T))
        zmesh = np.hstack((zmesh,np.array([zmesh[:,0]]).T))
        Mesh = np.hstack((Mesh,np.array([Mesh[:,0]],dtype=int).T))
        
        xsect.nodeDict = nodeDict
        laminate.mesh = Mesh
        laminate.xmesh = xmesh
        laminate.ymesh = ymesh
        laminate.zmesh = zmesh
        
        EIDmesh = np.zeros((np.size(xmesh,axis=0)-1,np.size(xmesh,axis=1)-1),dtype=int)
        
        for i in range(0,np.size(xmesh,axis=0)-1):
            for j in range(0,np.size(xmesh,axis=1)-1):
                newEID = int(max(elemDict.keys())+1)
                NIDs = [Mesh[i+1,j],Mesh[i+1,j+1],Mesh[i,j+1],Mesh[i,j]]
                nodes = [xsect.nodeDict[NID] for NID in NIDs]
                elemDict[newEID] = CQUADX(newEID,nodes,MID,matlib)
                EIDmesh[i,j] = newEID
        
        xsect.elemDict = elemDict
        ylen = np.size(xmesh,axis=0)-1
        xlen = np.size(xmesh,axis=1)-1
        laminate.plotx = np.zeros((ylen*2,xlen*2))
        laminate.ploty = np.zeros((ylen*2,xlen*2))
        laminate.plotz = np.zeros((ylen*2,xlen*2))
        laminate.plotc = np.zeros((ylen*2,xlen*2))
        laminate.EIDmesh = EIDmesh
        
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
class XSect:
    """Creates a beam cross-section object,
    
    This cross-section can be made of multiple materials which can be in
    general anisotropic. This is the main workhorse within the structures
    library.
    
    :Attributes:
    
    - `Color (touple)`: A length 3 touple used to define the color of the
        cross-section.
    - `Airfoil (obj)`: The airfoil object used to define the OML of the cross-
        section.
    - `typeXSect (str)`: Defines what type of cross-section is to be used.
        Currently the only supported type is 'box'.
    - `normalVector (1x3 np.array[float])`: Expresses the normal vector of the
        cross-section.
    - `nodeDict (dict)`: A dictionary of all nodes used to descretize the
        cross-section surface. The keys are the NIDs and the values stored
        are the Node objects.
    - `elemDict (dict)`: A dictionary of all elements used to descretize the
        cross-section surface. the keys are the EIDs and the values stored
        are the element objects.
    - `X (ndx6 np.array[float])`: A very large 2D array. This is one of the
        results of the cross-sectional analysis. This array relays the
        force and moment resultants applied to the cross-section to the
        nodal warping displacements exhibited by the cross-section.
    - `Y (6x6 np.array[float])`: This array relays the force and moment
        resultants applied to the cross-section to the rigid section
        strains and curvatures exhibited by the cross-section.
    - `dXdz (ndx6 np.array[float])`: A very large 2D array. This is one of the
        results of the cross-sectional analysis. This array relays the
        force and moment resultants applied to the cross-section to the
        gradient of the nodal warping displacements exhibited by the
        cross-section with respect to the beam axis.
    - `xt (float)`: The x-coordinate of the tension center (point at which
        tension and bending are decoupled)
    - `yt (float)`: The y-coordinate of the tension center (point at which
        tension and bending are decoupled)
    - `xs (float)`: The x-coordinate of the shear center (point at which shear
        and torsion are decoupled)
    - `ys (float)`: The y-coordinate of the shear center (point at which shear
        and torsion are decoupled)
    - `refAxis (3x1 np.array[float])`: A column vector containing the reference
        axis for the beam.
    - `bendAxes (2x3 np.array[float])`: Contains two row vectors about which
        bending from one axis is decoupled from bending about the other.
    - `F_raw (6x6 np.array[float])`: The 6x6 compliance matrix that results
        from cross-sectional analysis. This is the case where the reference
        axis is at the origin.
    - `K_raw (6x6 np.array[float])`: The 6x6 stiffness matrix that results
        from cross-sectional analysis. This is the case where the reference
        axis is at the origin.
    - `F (6x6 np.array[float])`: The 6x6 compliance matrix for the cross-
        section about the reference axis. The reference axis is by default
        at the shear center.
    - `K (6x6 np.array[float])`: The 6x6 stiffness matrix for the cross-
        section about the reference axis. The reference axis is by default
        at the shear center.
    - `T1 (3x6 np.array[float])`: The transformation matrix that converts
        strains and curvatures from the local xsect origin to the reference
        axis.
    - `T2 (3x6 np.array[float])`: The transformation matrix that converts
        forces and moments from the local xsect origin to the reference
        axis.
    - `x_m (1x3 np.array[float])`: Center of mass of the cross-section about in
        the local xsect CSYS
    - `M (6x6 np.array[float])`: This mass matrix relays linear and angular
        velocities to linear and angular momentum of the cross-section.
        

    :Methods:
    
    - `resetResults`: This method resets all results (displacements, strains
        and stresse) within the elements used by the cross-section object.
    - `calcWarpEffects`: Given applied force and moment resultants, this method
        calculates the warping displacement, 3D strains and 3D stresses
        within the elements used by the cross-section.
    - `printSummary`: This method is used to print characteristic attributes of
        the object. This includes the elastic, shear and mass centers, as
        well as the stiffness matrix and mass matrix.
    - `plotRigid`: This method plots the rigid cross-section shape, typically
        in conjunction with a full beam model.
    - `plotWarped`: This method plots the warped cross-section including a
        contour criteria, typically in conjuction with the results of the
        displacement of a full beam model.
        
    """
    def __init__(self,XID,Airfoil,xdim,laminates,matlib,**kwargs):
        """Instantiates a cross-section object.
        
        The constructor for the class is effectively responsible for creating
        the 2D desretized mesh of the cross-section. It is important to note
        that while meshing technically occurs in the constructor, the work is
        handeled by another class altogether. While not
        computationally heavily intensive in itself, it is responsible for
        creating all of the framework for the cross-sectional analysis.
        
        :Args:
        
        - `XID (int)`: The cross-section integer identifier.
        - `Airfoil (obj)`: An airfoil object used to determine the OML shape of
            the cross-section.
        - `xdim (1x2 array[float])`: The non-dimensional starting and stoping
            points of the cross-section. In other words, if you wanted to
            have your cross-section start at the 1/4 chord and run to the
            3/4 chord of your airfoil, xdim would look like xdim=[0.25,0.75]
        - `laminates (1xN array[obj])`: Laminate objects used to create the
            descretized mesh surface. Do not repeat a laminate within this
            array! It will referrence this object multiple times and not
            mesh the cross-section properly then!
        - `matlib (obj)`: A material library
        - `typeXSect (str)`: The general shape the cross-section should take.
            Note that currently only a box beam profile is supported.
            More shapes and the ability to add stiffeners to the
            cross-section will come in later updates.
        - `meshSize (int)`: The maximum aspect ratio you would like your 2D
            CQUADX elements to exhibit within the cross-section.
            
        :Returns:
        
        - None
            
        """
        #Save the cross-section ID
        self.XID = XID
        # Save the cross-section type:
        self.typeXSect = kwargs.pop('typeXSect','box')
        # Meshing aspect ratio
        meshSize = kwargs.pop('meshSize',4)
        # Initialize plotting color for the cross-section
        self.color = kwargs.pop('color',np.random.rand(3))
        elemX = kwargs.pop('elemX',1)
        elemY = kwargs.pop('elemY',1)
        MID = kwargs.pop('MID',1)
        elemOrder = kwargs.pop('elemOrder',1)
        nelem = kwargs.pop('nelem',8)
        # Save the airfoil object used to define the OML of the cross-section
        self.airfoil = Airfoil
        # Save the laminate array (and thus laminate objects) to be used
        self.laminates = laminates
        # Save the vector normal to the plane of the cross-section in the
        # global coordinate system
        self.normal_vector = np.array([0.,0.,1.])
        x0 = xdim[0]
        xf = xdim[1]
        
        mesher = Mesher()
        # Begin the meshing process for a box beam cross-section profile
        if self.typeXSect=='box':
            mesher.boxBeam(self,meshSize,x0,xf,matlib)
        elif self.typeXSect=='circle':
            r = kwargs.pop('r',self.airfoil.c)
            mesher.cylindricalTube(self,r,meshSize,x0,xf,matlib)
        elif self.typeXSect=='laminate':
            mesher.laminate(self, meshSize, x0, xf, matlib)
        elif self.typeXSect=='rectBox':
            mesher.rectBoxBeam(self, meshSize, x0, xf, matlib)
        elif self.typeXSect=='solidBox':
            mesher.solidBox(self,elemX, elemY, x0, xf, MID, matlib, elemOrder)
        elif self.typeXSect=='rectHole':
            mesher.rectangleHole(self, nelem, xdim[0], xdim[1], xdim[2], MID, matlib)
            
    def xSectionAnalysis(self,**kwargs):
        """Analyzes an initialized corss-section.
        
        This is the main workhorse of the class. This method assembles the
        finite element model generated using the meshing class, and solve the
        HIGH dimensional equilibrium equations associated with the cross-
        section. In doing so, it generates the warping displacement, the
        section strain, and the gradient of the warping displacement along the
        beam axis as a function of force-moment resultants. With these three
        things, the 3D strains->stresses can be recovered.
        
        This method has been EXTENSIVELY tested and validated against
        various sources (see theory guide for more info). Since this method
        is so robust, the biggest limitation of the XSect class is what the
        mesher is capable of meshing. Finally, keep in mind that due to the
        high dimensionality of this problem, this method uses up a lot of
        resources (primarily memory). If this method is taking too many
        resources, choose a larger aspect ratio for your XSect initialization.
        
        :Args:
        
        - `ref_ax (str or 1x2 array[float])`: Currently there are two supported
            input types for this class. The first is the are string key-words.
            These are 'shearCntr', 'massCntr', and 'origin'. Currently
            'shearCntr' is the default value. Also suported is the ability to
            pass a length 2 array containing the x and y coordinates of the
            reference axis relative to the origin. This would take the form of:
            ref_ax=[1.,3.] to put the reference axis at x,y = 1.,3.
            
        :Returns:
        
        - None
            
        """
        # Initialize the reference axis:
        ref_ax = kwargs.pop('ref_ax','shearCntr')
        # Create local reference to the node dictionary
        nodeDict = self.nodeDict
        # Create local reference to the element dictionary
        elemDict = self.elemDict
        # Initialize the D matrix, responsible for decoupling rigid cross-
        # section displacement from warping cross-section displacement
        nd = 3*len(nodeDict.keys())
        D = lil_matrix((6,nd), dtype=np.float64)
        for i in range(0,len(nodeDict.keys())):
            tmpNode = nodeDict[i]
            tempx = tmpNode.x[0]
            tempy = tmpNode.x[1]
            D[:,3*i:3*i+3] = lil_matrix(np.array([[1,0,0],\
                                       [0,1,0],\
                                       [0,0,1],\
                                       [0,0,tempy],\
                                       [0,0,-tempx],\
                                       [-tempy,tempx,0]]))
        D = D.T
        # Initialize Matricies used in solving the equilibruim equations:
        Tr =csr_matrix((6,6));Tr[0,4] = -1;Tr[1,3] = 1
        A = np.zeros((6,6))
        E = np.zeros((nd,nd))
        L = np.zeros((nd,6))
        R = np.zeros((nd,6))
        C = np.zeros((nd,nd))
        M = np.zeros((nd,nd))
        Z6 = csr_matrix((6,6))
        # Initialize the cross-section mass per unit length
        m = 0.
        # Initialize the first mass moment of inertia about x
        xm = 0.
        # Initialize the first mass moment of inertia about y
        ym = 0.
        #for i in range(0,len(elemDict.keys())):
        # For all elements in the cross-section mesh
        for EID, elem in elemDict.iteritems():
            #Select the element
            #tempElem = elemDict[i]
            # Get the NIDs reference by the element
            tempNodes = elem.NIDs
            # Update the cross-section mass
            m += elem.mass
            # Update the first mass moment of ineratia about x
            xm+= elem.mass*elem.x(0.,0.)
            # Update the first mass moment of ineratia about y
            ym+= elem.mass*elem.y(0.,0.)
            # If the 2D element is a CQUADX
            if (str(elem.type)=='CQUADX') or (str(elem.type)=='CQUADX9'):
                # Create local references to the element equilibrium matricies
                A += elem.Ae
                Re = elem.Re
                Ee = elem.Ee
                Ce = elem.Ce
                Le = elem.Le
                Me = elem.Me
                # Cross-section finite element matrix assembely
                for j in range(0,len(tempNodes)):
                    row = tempNodes[j]
                    R[3*row:3*row+3,:] = R[3*row:3*row+3,:] + Re[3*j:3*j+3,:]
                    L[3*row:3*row+3,:] = L[3*row:3*row+3,:] + Le[3*j:3*j+3,:]
                    for k in range(0,len(tempNodes)):
                        col = tempNodes[k]
                        E[3*row:3*row+3,3*col:3*col+3] = E[3*row:3*row+3,3*col:3*col+3] + Ee[3*j:3*j+3,3*k:3*k+3]
                        C[3*row:3*row+3,3*col:3*col+3] = C[3*row:3*row+3,3*col:3*col+3] + Ce[3*j:3*j+3,3*k:3*k+3]
                        M[3*row:3*row+3,3*col:3*col+3] = M[3*row:3*row+3,3*col:3*col+3] + Me[3*j:3*j+3,3*k:3*k+3]
            elif (str(elem.type)=='TRIA3'):
                #TODO: Still need to add a TRIA3 class
                pass
        # Cross-section matricies currently not saved to xsect object to save
        # memory.
#        self.A = A
#        self.R = R
#        self.E = E
#        self.C = C
#        self.L = L
#        self.M = M
#        self.D = D
        # Convert to sparse matricies:
        A = csr_matrix(A)
        R = csr_matrix(R)
        E = csr_matrix(E)
        C = csr_matrix(C)
        L = csr_matrix(L)
        M = csr_matrix(M)
        D = csr_matrix(D)
        #TrT = csc_matrix(Tr.T)
        #Z6 = csc_matrix(Z6)
        
        
                
        # SOLVING THE EQUILIBRIUM EQUATIONS
        # Assemble state matrix for first equation
        EquiA1 = vstack((hstack((E,R,D)),hstack((R.T,A,Z6)),\
                                        hstack((D.T,Z6,Z6))))
        # Assemble solution vector for first equation
        Equib1 = np.vstack((np.zeros((nd,6)),Tr.T.toarray(),Z6.toarray()))
        # LU factorize state matrix as it will be used twice
        #lu,piv = linalg.lu_factor(EquiA1)
        #del EquiA1
        # Solve system
        #sol1 = linalg.lu_solve((lu,piv),Equib1,check_finite=False)
        #del Equib1
        sol1_1 = np.matrix(minres(EquiA1,Equib1[:,0],tol=1e-10)[0]).T
        sol1_2 = np.matrix(minres(EquiA1,Equib1[:,1],tol=1e-10)[0]).T
        sol1_3 = np.matrix(minres(EquiA1,Equib1[:,2],tol=1e-10)[0]).T
        sol1_4 = np.matrix(minres(EquiA1,Equib1[:,3],tol=1e-10)[0]).T
        sol1_5 = np.matrix(minres(EquiA1,Equib1[:,4],tol=1e-10)[0]).T
        sol1_6 = np.matrix(minres(EquiA1,Equib1[:,5],tol=1e-10)[0]).T
        sol1 = np.hstack((sol1_1,sol1_2,sol1_3,sol1_4,sol1_5,sol1_6))
        # Recover gradient of displacement as a function of force and moment
        # resutlants
        dXdz = sol1[0:nd,:]
        self.dXdz = sol1[0:nd,:]
        # Save the gradient of section strains as a function of force and
        # moment resultants
        self.dYdz = sol1[nd:nd+6,:]
        # Set up the first of two solution vectors for second equation
        Equib2_1 = vstack((hstack((-(C-C.T),L))\
            ,hstack((-L.T,Z6)),csr_matrix((6,nd+6),dtype=np.float64)))
        # Set up the second of two solution vectors for second equation
        Equib2_2 = vstack((csr_matrix((nd,6),dtype=np.float64),eye(6,6),Z6))
        Equib2 = Equib2_1*csr_matrix(sol1[0:nd+6,:])+Equib2_2
        del Equib2_1
        del Equib2_2
        # Add solution vectors and solve second equillibrium equation
        sol2_1 = np.matrix(minres(EquiA1,Equib2[:,0].toarray(),tol=1e-10)[0]).T
        sol2_2 = np.matrix(minres(EquiA1,Equib2[:,1].toarray(),tol=1e-10)[0]).T
        sol2_3 = np.matrix(minres(EquiA1,Equib2[:,2].toarray(),tol=1e-10)[0]).T
        sol2_4 = np.matrix(minres(EquiA1,Equib2[:,3].toarray(),tol=1e-10)[0]).T
        sol2_5 = np.matrix(minres(EquiA1,Equib2[:,4].toarray(),tol=1e-10)[0]).T
        sol2_6 = np.matrix(minres(EquiA1,Equib2[:,5].toarray(),tol=1e-10)[0]).T
        
        sol2 = np.hstack((sol2_1,sol2_2,sol2_3,sol2_4,sol2_5,sol2_6))
        
        X = sol2[0:nd,0:6]
        # Store the warping displacement as a funtion of force and moment
        # resultants
        self.X = X
        # Store the section strain as a function of force and moment resultants
        Y = sol2[nd:nd+6,0:6]
        self.Y = Y
        #Solve for the cross-section compliance
        #comp1 = np.vstack((X,dXdz,Y))
        #comp2 = np.vstack((np.hstack((E,C,R)),np.hstack((C.T,M,L)),np.hstack((R.T,L.T,A))))
        #F = np.dot(comp1.T,np.dot(comp2,comp1))
        #del comp2
        Xcompr = csr_matrix(X)
        Ycompr = csr_matrix(Y)
        dXdzcompr = csr_matrix(dXdz)
        t1 = E*Xcompr+C*dXdzcompr+R*Ycompr
        t2 = C.T*Xcompr+M*dXdzcompr+L*Ycompr
        t3 = R.T*Xcompr+L.T*dXdzcompr+A*Ycompr
        F = Xcompr.T*t1+dXdzcompr.T*t2+Ycompr.T*t3
        F = F.toarray()
        # Store the compliance matrix taken about the xsect origin
        self.F_raw = F
        # Store the stiffness matrix taken about the xsect origin
        self.K_raw = np.linalg.inv(F)
        # Calculate the tension center
        self.xt = (-F[3,3]*F[4,2]+F[3,4]*F[3,2])/(F[3,3]*F[4,4]-F[3,4]**2)
        self.yt = (-F[3,2]*F[4,4]+F[3,4]*F[4,2])/(F[3,3]*F[4,4]-F[3,4]**2)
        # Calculate axis about which bedning is decoupled
        if np.abs(self.K_raw[3,4])<0.1:
            self.bendAxes = np.array([[1.,0.,0.,],[0.,1.,0.]])
        else:
            trash,axes = linalg.eig(np.array([[self.K_raw[3,3],self.K_raw[3,4]],\
                        [self.K_raw[4,3],self.K_raw[4,4]]]))
            self.bendAxes = np.array([[axes[0,0],axes[1,0],0.,],[axes[0,1],axes[1,1],0.]])
        # Calculate the location of the shear center neglecting the bending
        # torsion coupling contribution:
        
        # An error tolerance of 1% is chosen as the difference between shear
        # center locations at the beggining and end of the non-dimensional beam
        es = 1.
        z = 1.
        L = 1.
        xs = (-F[5,1]+F[5,3]*(L-z))/F[5,5]
        ys = (F[5,0]+F[5,4]*(L-z))/F[5,5]
        xsz0 = (-F[5,1]+F[5,3]*(L))/F[5,5]
        ysz0 = (F[5,0]+F[5,4]*(L))/F[5,5]
        eax = (xs-xsz0)/xs*100.
        eay = (ys-ysz0)/ys*100.
        if eax>es or eay>es:
            print('CAUTION: The shear center does not appear to be a cross-'\
                'section property, and will vary along the length of the beam.')
        self.xs = xs
        self.ys = ys
        # Calculate the mass center of the cross-section
        Ixx = 0.; Ixy=0.; Iyy=0;
        self.x_m = np.array([xm/m,ym/m,0.])
        # Establish reference axis location
        if ref_ax=='shearCntr':
            self.refAxis = np.array([[self.xs],[self.ys],[0.]])
            xref = -self.refAxis[0,0]
            yref = -self.refAxis[1,0]
        elif ref_ax=='massCntr':
            self.refAxis = np.array([[self.x_m[0]],[self.x_m[1]],[0.]])
            xref = -self.refAxis[0,0]
            yref = -self.refAxis[1,0]
        elif ref_ax=='origin':
            self.refAxis = np.array([[0.],[0.],[0.]])
            xref = -self.refAxis[0,0]
            yref = -self.refAxis[1,0]
        else:
            if len(ref_ax)==2:
                self.refAxis = np.array([[ref_ax[0]],[ref_ax[1]],[0.]])
                xref = -self.refAxis[0]
                yref = -self.refAxis[1]
            else:
                raise ValueError('You entered neither a supported reference axis'\
                'keyword, nor a valid length 2 array containing the x and y'\
                'beam axis reference coordinates for the cross-section.')
        # Strain reference axis transformation
        self.T1 = np.array([[1.,0.,0.,0.,0.,-yref],[0.,1.,0.,0.,0.,xref],\
            [0.,0.,1.,yref,-xref,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],\
            [0.,0.,0.,0.,0.,1.]])
        # Force reference axis transformation
        self.T2 = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],\
            [0.,0.,1.,0.,0.,0.],[0.,0.,-yref,1.,0.,0.],[0.,0.,xref,0.,1.,0.],\
            [yref,-xref,0.,0.,0.,1.]])
        self.F = np.dot(np.linalg.inv(self.T1),np.dot(self.F_raw,self.T2))
        self.K = np.dot(np.linalg.inv(self.T2),np.dot(self.K_raw,self.T1))
        # Reset all element cross-section matricies to free up memory
        for EID, elem in self.elemDict.iteritems():
            #elem.clearXSectionMatricies()
            # Initialize Guass points for integration
            etas = np.array([-1,1])*np.sqrt(3)/3
            xis = np.array([-1,1])*np.sqrt(3)/3
            # Calculate the second mass moments of inertia about the reference
            # axis
            for k in range(0,np.size(xis)):
                for l in range(0,np.size(etas)):
                    Jmat = elem.J(etas[l],xis[k])
                    Jdet = abs(np.linalg.det(Jmat))
                    # Add to the cross-section second mass moments of inertia
                    Ixx+=elem.rho*Jdet*(elem.y(etas[l],xis[k])-self.refAxis[1])**2
                    Iyy+=elem.rho*Jdet*(elem.x(etas[l],xis[k])-self.refAxis[0])**2
                    Ixy+=elem.rho*Jdet*(elem.y(etas[l],xis[k])-\
                        self.refAxis[1])*(elem.x(etas[l],xis[k])-self.refAxis[0])
        # Assemble cross-section mass matrix
        self.M = np.array([[m,0.,0.,0.,0.,-m*(self.x_m[1]-self.refAxis[1])],\
                           [0.,m,0.,0.,0.,m*(self.x_m[0]-self.refAxis[0])],\
                           [0.,0.,m,m*(self.x_m[1]-self.refAxis[1]),-m*(self.x_m[0]-self.refAxis[0]),0.],\
                           [0.,0.,m*(self.x_m[1]-self.refAxis[1]),Ixx,-Ixy,0.],\
                           [0.,0.,-m*(self.x_m[0]-self.refAxis[0]),-Ixy,Iyy,0.],\
                           [-m*(self.x_m[1]-self.refAxis[1]),m*(self.x_m[0]-self.refAxis[0]),0.,0.,0.,Ixx+Iyy]])
    def resetResults(self):
        """Resets displacements, stress and strains within an xsect
        
        This method clears all results (both warping, stress, and strain)
        within the elements in the xsect object.
        
        :Args:
        
        - None
            
        :Returns:
        
        - None
            
        """
        # For all elements within the cross-section
        for EID, elem in self.elemDict.iteritems():
            # Clear the results
            elem.resetResults()
    def calcWarpEffects(self,**kwargs):
        """Calculates displacements, stresses, and strains for applied forces
        
        The second most powerful method of the XSect class. After an analysis
        is run, the FEM class stores force and moment resultants within the
        beam element objects. From there, warping displacement, strain and
        stress can be determined within the cross-section at any given location
        within the beam using this method. This method will take a while though
        as it has to calculate 4 displacements and 24 stresses and strains for
        every element within the cross-section. Keep that in mind when you are
        surveying your beam or wing for displacements, stresses and strains.
        
        :Args:
        
        - `force (6x1 np.array[float])`: This is the internal force and moment
            resultant experienced by the cross-section.
            
        :Returns:
        
        - None
            
        """
        # Initialize the applied force
        frc = kwargs.pop('force',np.zeros((6,1)))
        frc = np.reshape(frc,(6,1))
        # Calculate the force applied at the origin of the cross-section
        th = np.dot(np.linalg.inv(self.T2),frc)
        # Generate nodal warping displacements
        u = np.dot(self.X,th)
        # Generate section strains
        strn0 = np.dot(self.Y,th)
        # Generate gradient of warping
        dudz = np.dot(self.dXdz,th)
        # For each element in the cross-section:
        for EID, elem in self.elemDict.iteritems():
            # Initialize the element warping vector for strain calc
            uelem = np.zeros((12,1))
            # Initialize the element warping grad vector for strain calc
            dudzelem = np.zeros((12,1))
            # For all nodes in the element
            for j in range(0,4):
                tmpNID = elem.NIDs[j]
                # Save warping displacement
                uelem[3*j:3*j+3] = u[3*tmpNID:3*tmpNID+3]
                # Save warping gradient
                dudzelem[3*j:3*j+3] = dudz[3*tmpNID:3*tmpNID+3]
            # Initialize strain vectors
            tmpEps = np.zeros((6,4))
            # Initialize stress vectors
            tmpSig = np.zeros((6,4))
            # Initialize Xis (strain sampling points)
            xis = np.array([-1,1,1,-1])#*np.sqrt(3.)/3.
            # Initialize Etas (strain sampling points)
            etas = np.array([-1,-1,1,1])#*np.sqrt(3.)/3.
            # Calculate Strain
            for j in range(0,4):
                # Initialize S:
                S = np.zeros((6,3));S[3,0]=1;S[4,1]=1;S[5,2]=1
                # Calculate Z at the corner:
                Z = elem.Z(etas[j],xis[j])
                # Calculate the Jacobian at the element corner:
                tmpJ = elem.J(etas[j],xis[j])
                # Calculate the inverse of the Jacobian
                Jmatinv = np.linalg.inv(tmpJ)
                # Initialize part of the strain displacement matrix
                Bxi = np.zeros((6,3))
                Bxi[0,0] = Bxi[2,1] = Bxi[3,2] = Jmatinv[0,0]
                Bxi[1,1] = Bxi[2,0] = Bxi[4,2] = Jmatinv[1,0]
                # Initialize part of the strain displacement matrix
                Beta = np.zeros((6,3))
                Beta[0,0] = Beta[2,1] = Beta[3,2] = Jmatinv[0,1]
                Beta[1,1] = Beta[2,0] = Beta[4,2] = Jmatinv[1,1]
                # Assemble the full strain displacement matrix
                BN = np.dot(Bxi,elem.dNdxi(etas[j])) + np.dot(Beta,elem.dNdeta(xis[j]))
                # Initialize shape function displacement matrix
                N = elem.N(etas[j],xis[j])
                # Calculate the 3D strain state
                tmpEps[:,j] = np.transpose(np.dot(S,np.dot(Z,strn0))+\
                    np.dot(BN,uelem)+\
                    np.dot(S,np.dot(N,dudzelem)))
                # Calculate the 3D stress state in the cross-section CSYS
                tmpSig[:,j] = np.dot(elem.Q,tmpEps[:,j])
            # Save the displacement vector of the element nodes
            elem.U = uelem
            # Save the strain states at all 4 corners for the element
            elem.Eps = tmpEps
            # Save the stress states at all 4 corners for the element
            elem.Sig = tmpSig    
            # Save the forces applied to the beam nodes
                        
    def plotXSect(self,**kwargs):
        """Plots an XSect object.
        
        Intended primarily as a private method for new cross-section mesh
        debugging but left public, this method will plot all of the laminate
        meshes within the cross-section.
        
        :Args:
        
        - `figName (str)`: The name of the figure.
            
        :Returns:
        
        - `(fig)`: Will plot a mayavi figure
            
        """
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        mlab.figure(figure=figName)
        for laminate in self.laminates:
            laminate.plotLaminate(figName=figName)
    def printSummary(self,refAxis=True,decimals=8,**kwargs):
        """Print characterisic information about the cross-section.
        
        This method prints out characteristic information about the cross-
        section objects. By default, the method will print out the location of
        the reference axis, the shear, tension, and mass center. This method
        if requested will also print the stiffness and mass matricies.
        
        :Args:
        
        - `refAxis (bool)`: Boolean to determine if the stiffness matrix
            printed should be about the reference axis (True) or about the
            local xsect origin (False).
        - `stiffMat (bool)`: Boolean to determine if the stiffness matrix
            should be printed.
        - `tensCntr (bool)`: Boolean to determine if the location of the tension
            center should be printed.
        - `shearCntr (bool)`: Boolean to determine if the location of the shear
            center should be printed.
        - `massCntr (bool)`: Boolean to determine if the location of the mass
            center should be printed.
        - `refAxisLoc (bool)`: Boolean to determine if the location of the
            reference axis should be printed.
        
        :Returns:
        
        - `(str)`: Prints out a string of information about the cross-section.
        
        """
        # Print xsect info:
        print('XSect: %d' %(self.XID))
        print('Type of cross-section is: '+self.typeXSect)
        print('The OML selected is: '+self.airfoil.name)
        # Print the 6x6 stiffnes matrix?
        stiffMat = kwargs.pop('stiffMat',False)
        # Print tension center?
        tensCntr = kwargs.pop('tensCntr',True)
        # Print shear center?
        shearCntr = kwargs.pop('shearCntr',True)
        # Print mass matrix?
        massMat = kwargs.pop('massMat',False)
        # Print mass center?
        massCntr = kwargs.pop('massCntr',True)
        # Print reference axis?
        refAxisLoc = kwargs.pop('refAxis',True)
        
        if refAxisLoc:
            print('The x-coordinate of the reference axis is: %4.6f' %(self.refAxis[0]))
            print('The y-coordinate of the reference axis is: %4.6f' %(self.refAxis[1]))
        if massCntr:
            print('The x-coordinate of the mass center is: %4.6f' %(self.x_m[0]))
            print('The y-coordinate of the mass center is: %4.6f' %(self.x_m[1]))
        if shearCntr:
            print('The x-coordinate of the shear center is: %4.6f' %(self.xs))
            print('The y-coordinate of the shear center is: %4.6f' %(self.ys))
        if tensCntr:
            print('The x-coordinate of the tension center is: %4.6f' %(self.xt))
            print('The y-coordinate of the tension center is: %4.6f' %(self.yt))
        if stiffMat:
            if refAxis:
                print('\n\nThe cross-section stiffness matrix about the reference axis is:')
                print(tabulate(np.around(self.K,decimals=decimals),tablefmt="fancy_grid"))
            else:
                print('\n\nThe cross-section stiffness matrix about the xsect origin is:')
                print(tabulate(np.around(self.K_raw,decimals=decimals),tablefmt="fancy_grid"))
        if massMat:
            print('\n\nThe cross-section mass matrix about the reference axis is:')
            print(tabulate(np.around(self.M,decimals=decimals),tablefmt="fancy_grid"))
    def plotRigid(self,**kwargs):
        """Plots the rigid cross-section along a beam.
        
        This method is very useful for visually debugging a structural model.
        It will plot out the rigid cross-section in 3D space with regards to
        the reference axis.
        
        :Args:
        
        - `x (1x3 np.array[float])`: The rigid location on your beam you are
            trying to plot:
        - `beam_axis (1x3 np.array[float])`: The vector pointing in the
            direction of your beam axis.
        - `figName (str)`: The name of the figure.
        - `wireMesh (bool)`: A boolean to determine of the wiremesh outline
            should be plotted.*
            
        :Returns:
        
        - `(fig)`: Plots the cross-section in a mayavi figure.
            
        .. Note:: Because of how the mayavi wireframe keyword works, it will
        apear as though the cross-section is made of triangles as opposed to
        quadrilateras. Fear not! They are made of quads, the wireframe is just
        plotted as triangles.
        
        """
        # The rigid translation of the cross-section
        x = kwargs.pop('x',np.array([0.,0.,0.]))
        # The rotation matrix mapping the cross-section from the local frame to
        # the global frame
        RotMat = kwargs.pop('RotMat',np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
        # The figure name
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Whether the wiremesh should be plotted
        wireMesh = kwargs.pop('mesh',True)
        # Initialize which figure is being plotted on
        mlab.figure(figure=figName)
        # For all laminate objects in the cross-section
        for lam in self.laminates:
            # Initialize xyz array's to be plotted
            plotx = np.copy(lam.xmesh)
            ploty = np.copy(lam.ymesh)
            plotz = np.copy(lam.zmesh)
            # Rotate all of the points within the arrays
            for i in range(0,np.size(lam.mesh,axis=0)):
                for j in range(0,np.size(lam.mesh,axis=1)):
                    tmpPos = np.array([[plotx[i,j]],[ploty[i,j]],[plotz[i,j]]])
                    newPos = np.dot(RotMat,tmpPos)
                    plotx[i,j] = newPos[0]
                    ploty[i,j] = newPos[1]
                    plotz[i,j] = newPos[2]
            # Plot the three xyz arrays
            mlab.mesh(plotx+x[0]-self.refAxis[0],ploty+x[1]-self.refAxis[1],plotz+x[2],color=tuple(self.color))
            # Plot the wireframe if desired.
            if wireMesh:
                mlab.mesh(plotx+x[0]-self.refAxis[0],ploty+x[1]-self.refAxis[1],plotz+x[2],\
                    representation='wireframe',color=(0,0,0))
    def plotWarped(self,**kwargs):
        """Plots the warped cross-section along a beam.
        
        Once an analysis has been completed, this method can be utilized in
        order to plot the results anywhere along the beam.
        
        :Args:
        
        - `displScale (float)`: The scale by which all rotations and
            displacements will be mutliplied in order make it visually
            easier to detect displacements.
        - `x (1x3 np.array[float])`: The rigid location on your beam you are
            trying to plot:
        - `U (1x6 np.array[float])`: The rigid body displacements and rotations
            experienced by the cross-section.
        - `beam_axis (1x3 np.array[float])`: The vector pointing in the
            direction of your beam axis.
        - `contour (str)`: Determines what value is to be plotted during as a
            contour in the cross-section.
        - `figName (str)`: The name of the figure.
        - `wireMesh (bool)`: A boolean to determine of the wiremesh outline
            should be plotted.*
        - `contLim (1x2 array[float])`: Describes the upper and lower bounds of
            contour color scale.
        - `warpScale (float)`: The scaling factor by which all warping
            displacements in the cross-section will be multiplied.
            
        :Returns:
        
        - `(fig)`: Plots the cross-section in a mayavi figure.
            
        .. Note:: Because of how the mayavi wireframe keyword works, it will
        apear as though the cross-section is made of triangles as opposed to
        quadrilateras. Fear not! They are made of quads, the wireframe is just
        plotted as triangles.
        
        """
        # INPUT ARGUMENT INITIALIZATION
        # Select Displacement Scale
        displScale = kwargs.pop('displScale',1.)
        # The rigid translation of the cross-section
        x = kwargs.pop('x',np.zeros(3))
        # The defomation (tranltation and rotation) of the beam node and cross-section
        U = displScale*kwargs.pop('U',np.zeros(6))
        # The rotation matrix mapping the cross-section from the local frame to
        # the global frame
        RotMat = kwargs.pop('RotMat',np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
        # The figure name
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Show a contour
        contour = kwargs.pop('contour','VonMis')
        # Show wire mesh?
        wireMesh = kwargs.pop('mesh',False)
        # Stress Limits
        contLim = kwargs.pop('contLim',[])
        # Establish the warping scaling factor
        warpScale = kwargs.pop('warpScale',1.)
        # Establish if the colorbar should be generated:
        colorbar = kwargs.pop('colorbar',True)
        plots = kwargs.pop('plots',[])
        # Initialize on what figure the cross-section is to be plotted
        mlab.figure(figure=figName)
        # Create a rotation helper
        rh = RotationHelper()
        # Rotate the rotations from the global frame to the local frame:
        UlocalRot = np.dot(RotMat.T,np.reshape(U[3:6],(3,1)))
        # Calculate the rotation matrix about the local z axis
        RotZ = rh.rotXYZ(np.array([0.,0.,UlocalRot[2]]),deg2rad=False)
        # Calculate the rotation matrix about the local x-axis
        RotX = rh.rotXYZ(np.array([UlocalRot[0],0.,0.]),deg2rad=False)
        # Calculate the rotation matrix about the local y-axis
        RotY = rh.rotXYZ(np.array([0.,UlocalRot[1],0.]),deg2rad=False)
        # Create local reference of the reference axis
        refAxis = self.refAxis
        # Create local reference to the shear center locations
        xsc = self.xs
        ysc = self.ys
        # Create local reference to the tension center locations
        xtc = self.xt
        ytc = self.yt
        # Add the warping displacements to original xyz coordinates
        for lam in self.laminates:
            eidArray = lam.EIDmesh
            lamxsize = np.size(eidArray,axis=0)
            lamysize = np.size(eidArray,axis=1)
            # Initialize the x,y,z and contour array's to be plotted
            plotx = np.zeros((2*lamxsize,2*lamysize))
            ploty = np.zeros((2*lamxsize,2*lamysize))
            plotz = np.zeros((2*lamxsize,2*lamysize))
            plotc = np.zeros((2*lamxsize,2*lamysize))
            # For all elements in the laminate
            for i in range(0,lamxsize):
                for j in range(0,lamysize):
                    tmpEID = eidArray[i,j]
                    elem = self.elemDict[tmpEID]
                    xdef,ydef,zdef = elem.getDeformed(warpScale=warpScale)
                    plotx[2*i:2*i+2,2*j:2*j+2] = xdef
                    ploty[2*i:2*i+2,2*j:2*j+2] = ydef
                    plotz[2*i:2*i+2,2*j:2*j+2] = zdef
                    plotc[2*i:2*i+2,2*j:2*j+2] = elem.getStressState(crit=contour)
            # Translate the cross-section points to the shear center
            #plotx = plotx-xsc
            #ploty = ploty-ysc
            # Conduct torsion rotation about the shear center
            for i in range(0,np.size(plotx,axis=0)):
                for j in range(0,np.size(plotx,axis=1)):
                    # Establish the temporary position vector and translate to
                    # shear center
                    tmpPos = np.array([[plotx[i,j]-xsc],[ploty[i,j]-ysc],[plotz[i,j]]])
                    # Apply torsion rotation and translate to tension center
                    tmpPos = np.dot(RotZ,tmpPos)-np.array([[xtc-xsc],[ytc-ysc],[0.]])
                    # Apply moment rotations and translate to reference axis
                    tmpPos = np.dot(RotY,np.dot(RotX,tmpPos))\
                        -np.array([[refAxis[0]-xtc],[refAxis[1]-ytc],[0.]])
                    # Apply rotation to global frame
                    newPos = np.dot(RotMat,tmpPos)
                    # Add rotated points back
                    plotx[i,j] = newPos[0]
                    ploty[i,j] = newPos[1]
                    plotz[i,j] = newPos[2]
            # Plot the laminate surface
            if isinstance(contour,str):
                if len(contLim)==0:
                    surf = mlab.mesh(plotx+x[0]+U[0],ploty+x[1]+U[1],plotz+x[2]+U[2],\
                        scalars=plotc)
                else:
                    surf = mlab.mesh(plotx+x[0]+U[0],ploty+x[1]+U[1],plotz+x[2]+U[2],scalars=plotc,\
                        vmin=contLim[0],vmax=contLim[1])
                if colorbar:
                    mlab.colorbar()
            else:
                surf = mlab.mesh(plotx+x[0]+U[0],ploty+x[1]+U[1],plotz+x[2]+U[2],color=tuple(self.color))
            if wireMesh:
                mesh = mlab.mesh(plotx+x[0]+U[0],ploty+x[1]+U[1],plotz+x[2]+U[2],\
                    representation='wireframe',color=tuple(self.color[::-1]))
                plots += [mesh]
            plots += [surf]
            print('Max Criteria: %7.3f'% np.max(plotc))
            print('Min Criteria: %7.3f'% np.min(plotc))
class Beam(object):
    """The parent class for all beams finite elements.
    
    This class exists primarily to cut down on code repetition for all beam
    finite element objects. Of the two beam finite element classes, only one is
    currently supported and validated (being the TBeam class).
    
    :Attributes:
    
    - `U1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the first node.
    - `U2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the second node.
    - `Umode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the first node associated with the
        particular mode.
    - `Umode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the second node associated with the
        particular mode.
    - `F1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the first node.
    - `F2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the second node.
    - `Fmode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the first node associated with the
        particular mode.*
    - `Fmode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the second node associated with the
        particular mode.*
    - `xsect (obj)`: The cross-section object used to determine the beams
        stiffnesses.
    - `EID (int)`: The element ID of the beam.
    - `SBID (int)`: The associated Superbeam ID the beam object belongs to.
    - `n1 (obj)`: The first nodal object used by the beam.
    - `n2 (obj)`: The second nodal object used by the beam.
    - `Fe (12x1 np.array[float])`: The distributed force vector of the element
    - `Ke (12x12 np.array[float])`: The stiffness matrix of the beam.
    - `Keg (12x12 np.array[float])`: The geometric stiffness matrix of the
        beam. Used for beam buckling calculations.
    - `Me (12x12 np.array[float])`: The mass matrix of the beam.
    - `analysis_names (array[str])`: An array containing all of the string
        names being used as keys in either U1,U2,F1,F2,Umode1,Umode2,Fmode1
        Fmode2
        
    :Methods:
    
    - `printSummary`: This method prints out characteristic attributes of the
        beam finite element.
        
    .. Note:: The force and moments in the Fmode1 and Fmode2 could be completely
        fictitious and be left as an artifact to fascilitate plotting of warped
        cross-sections. DO NOT rely on this information being meaningful.
        
    """
    #TODO: Add some kind of print results methods for both printing to command
    # line and saving to a file.
    def __init__(self,xsect,EID,SBID):
        """Initializes a blank beam object.
        
        This method initializes attributes that are shared by all of the
        possible beam elements.
        
        :Args:
        
        - `xsect (obj)`: The cross-section object used by the beam.
        - `EID (int)`: The integer identifier of the beam element.
        - `SBID (int)`: The associated superbeam ID
            
        :Returns:
        
        - None
            
        """
        # Nodal displacement dictionary
        self.U1 = {}
        self.U2 = {}
        # Nodal displacements for eigenvalue solutions with multiple modes
        self.Umode1 = {}
        self.Umode2 = {}
        # Nodal force and moment resultant dictionary
        self.F1 = {}
        self.F2 = {}
        # Nodal force and moment resultant for eigenvalue solutions with
        # multiple modes
        self.Fmode1 = {}
        self.Fmode2 = {}
        # The beam's cross-section
        self.xsect = xsect
        # INITIALIZE ID's
        # The element ID
        self.EID = EID
        # The xsect ID
        self.XID = xsect.XID
        # The beam super-element ID
        self.SBID = SBID
        # Initialize Stifness and force matricies/vectors
        # Element force vector
        self.Fe = np.zeros((12,1),dtype=float)
        self.Ke = np.zeros((12,12),dtype=float)
        self.Keg = np.zeros((12,12),dtype=float)
        self.Me = np.zeros((12,12),dtype=float)
        self.T = np.zeros((12,12),dtype=float)
        # Initialize a dictionary to hold analysis names
        self.analysis_names = []
    def printSummary(self,decimals=8,**kwargs):
        """Prints out characteristic information about the beam element.
        
        This method by default prints out the EID, XID, SBID and the NIDs along
        with the nodes associated coordinates. Upon request, it can also print
        out the beam element stiffness, geometric stiffness, mass matricies and
        distributed force vector.
        
        :Args:
        
        - `nodeCoord (bool)`: A boolean to determine if the node coordinate
            information should also be printed.
        - `Ke (bool)`: A boolean to determine if the element stiffness matrix
            should be printed.
        - `Keg (bool)`: A boolean to determine if the element gemoetric
            stiffness matrix should be printed.
        - `Me (bool)`: A boolean to determine if the element mass matrix
            should be printed.
        - `Fe (bool)`: A boolean to determine if the element distributed force
            and moment vector should be printed.
            
        :Returns:
        
        - `(str)`: Printed summary of the requested attributes.
            
        """
        # Print the element ID
        print('Element: %d' %(self.EID))
        # Print the associated xsect ID
        XID = kwargs.pop('XID',False)
        # Print the associated superbeam ID
        SBID = kwargs.pop('SBID',False)
        # Determine if node coordinates should also be printed
        nodeCoord = kwargs.pop('nodeCoord',True)
        # Print the stiffness matrix
        Ke = kwargs.pop('Ke',False)
        # Print the geometric stiffness matrix
        Keg = kwargs.pop('Keg',False)
        # Print the mass matrix
        Me = kwargs.pop('Me',False)
        # Print the distributed force vector
        Fe = kwargs.pop('Fe',False)
        
        if XID:
            print('Cross-section: %d' %(self.XID))
        if SBID:
            print('Superbeam: %d' %(self.SBID))
        # Print the node information
        if nodeCoord:
            self.n1.printSummary()
            self.n2.printSummary()
        else:
            print('NID_1: %d' %(self.n1.NID))
            print('NID_2: %d' %(self.n2.NID))
        if Ke:
            print('The beam element stiffness matrix is:')
            print(tabulate(np.around(self.Ke,decimals=decimals)))
        if Keg:
            print('The beam element geometric stiffness matrix is:')
            print(tabulate(np.around(self.Keg,decimals=decimals)))
        if Me:
            print('The beam element mass matrix is:')
            print(tabulate(np.around(self.Me,decimals=decimals)))
        if Fe:
            print('The beam element force vector is:')
            print(tabulate(np.around(self.Fe,decimals=decimals)))
class EBBeam(Beam):
    """Euler-Bernoulli beam class.
    
    This class is currently unsuppoted. Please use the more accurace timoshenko
    beam class
    """
    def __init__(self,x1,x2,xsect,EID,SBID,nid1=1,nid2=2):
        #Description: Creases a single beam element, capable of being orientied
        #in any desired manner within 3d space.
        
        #INPUTS:
        #x1 - The x,y,z coordinate of the first node in the element
        #x2 - The x,y,z coordinate of the second node in the element
        #xsect - Cross-section object
        #eid - The element ID
        #   Note, the EID is inherited from the Superbeam EID if not otherwise specified
        #leid - The local element ID within a superbeam
        #nid1 - the first node ID
        #nid2 - the second node ID
        #Note, for now these elements can only sustain constant distributed loads
        Beam.__init__(self,xsect)
        self.type = 'EBbeam'
        self.n1 = Node(x1,nid1)
        self.n2 = Node(x2,nid2)
        h = np.sqrt((x2[0]-x1[0])**2+(x2[1]-x1[1])**2+(x2[2]-x1[2])**2)
        self.h = h
        self.xsect = xsect
        K = xsect.K
        #Lines below not needed, there for visual neatness
        C33 = K[2,2];C34 = K[2,3];C35 = K[2,4];C36 = K[2,5]
        C44 = K[3,3];C45 = K[3,4];C46 = K[3,5]
        C55 = K[4,4];C56 = K[4,5]
        C66 = K[5,5]
        ketmp = np.array([[12.*C44/h**3,12.*C45/h**3,0.,-6.*C44/h**2,-6.*C45/h**2,0.,-12.*C44/h**3,-12.*C45/h**3,0.,-6.*C44/h**2,-6.*C45/h**2,0.],\
                          [12.*C45/h**3,12.*C55/h**3,0.,-6.*C45/h**2,-6.*C55/h**2,0.,-12.*C45/h**3,-12.*C55/h**3,0.,-6.*C45/h**2,-6.*C55/h**2,0.],\
                          [0.,0.,C33/h,-C34/h,-C35/h,C36/h,0.,0.,-C33/h,C34/h,C35/h,-C36/h],\
                          [-6.*C44/h**2,-6.*C45/h**2,-C34/h,4.*C44/h,4.*C45/h,-C46/h,6.*C44/h**2,6.*C45/h**2,C34/h,2.*C44/h,2.*C45/h,C46/h],\
                          [-6.*C45/h**2,-6.*C55/h**2,-C35/h,4.*C45/h,4.*C55/h,-C56/h,6.*C45/h**2,6.*C55/h**2,C35/h,2.*C45/h,2.*C55/h,C56/h],\
                          [0.,0.,C36/h,-C46/h,-C56/h,C66/h,0.,0.,-C36/h,C46/h,C56/h,-C66/h],\
                          [-12.*C44/h**3,-12.*C45/h**3,0.,6.*C44/h**2,6.*C45/h**2,0.,12.*C44/h**3,12.*C45/h**3,0.,6.*C44/h**2,6.*C45/h**2,0.],\
                          [-12.*C45/h**3,-12.*C55/h**3,0.,6.*C45/h**2,6.*C55/h**2,0.,12.*C45/h**3,12.*C55/h**3,0.,6.*C45/h**2,6.*C55/h**2,0.],\
                          [0.,0.,-C33/h,C34/h,C35/h,-C36/h,0.,0.,C33/h,-C34/h,-C35/h,C36/h],\
                          [-6.*C44/h**2,-6.*C45/h**2,C34/h,2.*C44/h,2.*C45/h,C46/h,6.*C44/h**2,6.*C45/h**2,-C34/h,4.*C44/h,4.*C45/h,-C46/h],\
                          [-6.*C45/h**2,-6.*C55/h**2,C35/h,2.*C45/h,2.*C55/h,C56/h,6.*C45/h**2,6.*C55/h**2,-C35/h,4.*C45/h,4.*C55/h,-C56/h],\
                          [0.,0.,-C36/h,C46/h,C56/h,-C66/h,0.,0.,C36/h,-C46/h,-C56/h,C66/h]])            
        self.Ke = ketmp
        self.Fe = np.zeros((12,1),dtype=float)
        #Initialize the Geometric Stiffness Matrix
        kgtmp = np.array([[6./(5.*h),0.,0.,-1./10.,0.,0.,-6/(5.*h),0.,0.,-1./10.,0.,0.],\
                          [0.,6./(5.*h),0.,0.,-1./10.,0.,0.,-6./(5.*h),0.,0.,-1./10.,0.],\
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
                          [-1./10.,0.,0.,2.*h/15.,0.,0.,1./10.,0.,0.,-h/30.,0.,0.],\
                          [0.,-1./10.,0.,0.,2.*h/15.,0.,0.,1./10.,0.,0.,-h/30.,0.],\
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
                          [-6./(5.*h),0.,0.,1./10.,0.,0.,6./(5.*h),0.,0.,1./10.,0.,0.],\
                          [0.,-6./(5.*h),0.,0.,1./10.,0.,0.,6./(5.*h),0.,0.,1./10.,0.],\
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],\
                          [-1./10.,0.,0.,-h/30.,0.,0.,1./10.,0.,0.,2.*h/15.,0.,0.],\
                          [0.,-1./10.,0.,0.,-h/30.,0.,0.,1./10.,0.,0.,2.*h/15.,0.],\
                          [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])
        self.Keg = kgtmp
        
class TBeam(Beam):
    """Creates a Timoshenko beam finite element object.
    
    The primary beam finite element used by AeroComBAT, this beam element is
    similar to the Euler-Bernoulli beam finite element most are farmiliar with,
    with the exception that it has the ability to experience shear deformation
    in addition to just bending.
    
    :Attributes:
    
    - `type (str)`:String describing the type of beam element being used.
    - `U1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the first node.
    - `U2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 displacements and
        3 rotations at the second node.
    - `Umode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the first node associated with the
        particular mode.
    - `Umode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        displacements and rotations at the second node associated with the
        particular mode.
    - `F1 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the first node.
    - `F2 (dict)`: This dictionary contains the results of an analysis set. The
        keys are the string names of the analysis and the values stored are
        6x1 np.array[float] vectors containing the 3 internal forces and
        3 moments at the second node.
    - `Fmode1 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the first node associated with the
        particular mode.*
    - `Fmode2 (dict)`: This dictionary contains the results of a modal analysis
        set. The keys are the string names of the analysis and the values
        stored are 6xN np.array[float]. The columns of the array are the
        forces and moments at the second node associated with the
        particular mode.*
    - `xsect (obj)`: The cross-section object used to determine the beams
        stiffnesses.
    - `EID (int)`: The element ID of the beam.
    - `SBID (int)`: The associated Superbeam ID the beam object belongs to.
    - `n1 (obj)`: The first nodal object used by the beam.
    - `n2 (obj)`: The second nodal object used by the beam.
    - `Fe (12x1 np.array[float])`: The distributed force vector of the element
    - `Ke (12x12 np.array[float])`: The stiffness matrix of the beam.
    - `Keg (12x12 np.array[float])`: The geometric stiffness matrix of the
        beam. Used for beam buckling calculations.
    - `Me (12x12 np.array[float])`: The mass matrix of the beam.
    - `h (float)`: The magnitude length of the beam element.
    - `xbar (float)`: The unit vector pointing in the direction of the rigid
        beam.
    - `T (12x12 np.array[float])`:
        
    :Methods:
    
    - `printSummary`: This method prints out characteristic attributes of the
        beam finite element.
    - `plotRigidBeam`: Plots the the shape of the rigid beam element.
    - `plotDisplBeam`: Plots the deformed shape of the beam element.
    - `printInternalForce`: Prints the internal forces of the beam element for
        a given analysis set
        
    .. Note:: The force and moments in the Fmode1 and Fmode2 could be completely
    fictitious and be left as an artifact to fascilitate plotting of warped
    cross-sections. DO NOT rely on this information being meaningful.
    
    """
    def __init__(self,EID,x1,x2,xsect,SBID=0,nid1=0,nid2=1,chordVec=np.array([1.,0.,0.])):
        """Instantiates a timoshenko beam element.
        
        This method instatiates a finite element timoshenko beam element.
        Currently the beam must be oriented along the global y-axis, however
        full 3D orientation support for frames is in progress.
        
        :Args:
        
        - `x1 (1x3 np.array[float])`: The 3D coordinates of the first beam
            element node.
        - `x2 (1x3 np.array[float])`: The 3D coordinates of the second beam
            element node.
        - `xsect (obj)`: The cross-section object used to determine stiffnes
            and mass properties for the beam.
        - `EID (int)`: The integer identifier for the beam.
        - `SBID (int)`: The associated superbeam ID.
        - `nid1 (int)`: The first node ID
        - `nid2 (int)`: The second node ID
            
        :Returns:
        
        - None
            
        """
        # Inherit from Beam class
        Beam.__init__(self,xsect,EID,SBID)
        # Initialize element type
        self.type = 'Tbeam'
        # Verify properly dimensionalized coordinates are used to create the
        # nodes.
        if (len(x1) != 3) or (len(x2) != 3):
            raise ValueError('The nodal coordinates of the beam must be 3 dimensional.')
        # Create the node objects
        self.n1 = Node(nid1,x1)
        self.n2 = Node(nid2,x2)
        # Solve for the length of the beam
        h = np.linalg.norm(x2-x1)
        self.h = h
        # Solve for the beam unit vector
        self.xbar = (x2-x1)/h
        # Determine the Transformation Matrix
        zVec = self.xbar
        yVec = np.cross(zVec,chordVec)/np.linalg.norm(np.cross(zVec,chordVec))
        xVec = np.cross(yVec,zVec)/np.linalg.norm(np.cross(yVec,zVec))
        Tsubmat = np.vstack((xVec,yVec,zVec))
        self.T[0:3,0:3] = Tsubmat
        self.T[3:6,3:6] = Tsubmat
        self.T[6:9,6:9] = Tsubmat
        self.T[9:12,9:12] = Tsubmat
        self.xsect = xsect
        # Create a local reference to the cross-section stiffness matrix
        K = xsect.K
        # Lines below not needed, there for visual neatness
        C11 = K[0,0];C12 = K[0,1];C13 = K[0,2];C14 = K[0,3];C15 = K[0,4];C16 = K[0,5]
        C22 = K[1,1];C23 = K[1,2];C24 = K[1,3];C25 = K[1,4];C26 = K[1,5]
        C33 = K[2,2];C34 = K[2,3];C35 = K[2,4];C36 = K[2,5]
        C44 = K[3,3];C45 = K[3,4];C46 = K[3,5]
        C55 = K[4,4];C56 = K[4,5]
        C66 = K[5,5]
        # Initialize the Element Stiffness Matrix
        self.Kel = np.array([[C11/h,C12/h,C13/h,-C12/2+C14/h,C11/2+C15/h,C16/h,-C11/h,-C12/h,-C13/h,-C12/2-C14/h,C11/2-C15/h,-C16/h],\
                          [C12/h,C22/h,C23/h,-C22/2+C24/h,C12/2+C25/h,C26/h,-C12/h,-C22/h,-C23/h,-C22/2-C24/h,C12/2-C25/h,-C26/h],\
                          [C13/h,C23/h,C33/h,-C23/2+C34/h,C13/2+C35/h,C36/h,-C13/h,-C23/h,-C33/h,-C23/2-C34/h,C13/2-C35/h,-C36/h],\
                          [-C12/2+C14/h,-C22/2+C24/h,-C23/2+C34/h,-C24+C44/h+C22*h/4,C14/2-C25/2+C45/h-C12*h/4,-C26/2+C46/h,C12/2-C14/h,C22/2-C24/h,C23/2-C34/h,-C44/h+C22*h/4,C14/2+C25/2-C45/h-C12*h/4,C26/2-C46/h],\
                          [C11/2+C15/h,C12/2+C25/h,C13/2+C35/h,C14/2-C25/2+C45/h-C12*h/4,C15+C55/h+C11*h/4,C16/2+C56/h,-C11/2-C15/h,-C12/2-C25/h,-C13/2-C35/h,-C14/2-C25/2-C45/h-C12*h/4,-C55/h+C11*h/4,-C16/2-C56/h],\
                          [C16/h,C26/h,C36/h,-C26/2+C46/h,C16/2+C56/h,C66/h,-C16/h,-C26/h,-C36/h,-C26/2-C46/h,C16/2-C56/h,-C66/h],\
                          [-C11/h,-C12/h,-C13/h,C12/2-C14/h,-C11/2-C15/h,-C16/h,C11/h,C12/h,C13/h,C12/2+C14/h,-C11/2+C15/h,C16/h],\
                          [-C12/h,-C22/h,-C23/h,C22/2-C24/h,-C12/2-C25/h,-C26/h,C12/h,C22/h,C23/h,C22/2+C24/h,-C12/2+C25/h,C26/h],\
                          [-C13/h,-C23/h,-C33/h,C23/2-C34/h,-C13/2-C35/h,-C36/h,C13/h,C23/h,C33/h,C23/2+C34/h,-C13/2+C35/h,C36/h],\
                          [-C12/2-C14/h,-C22/2-C24/h,-C23/2-C34/h,-C44/h+C22*h/4,-C14/2-C25/2-C45/h-C12*h/4,-C26/2-C46/h,C12/2+C14/h,C22/2+C24/h,C23/2+C34/h,C24+C44/h+C22*h/4,-C14/2+C25/2+C45/h-C12*h/4,C26/2+C46/h],\
                          [C11/2-C15/h,C12/2-C25/h,C13/2-C35/h,C14/2+C25/2-C45/h-C12*h/4,-C55/h+C11*h/4,C16/2-C56/h,-C11/2+C15/h,-C12/2+C25/h,-C13/2+C35/h,-C14/2+C25/2+C45/h-C12*h/4,-C15+C55/h+C11*h/4,-C16/2+C56/h],\
                          [-C16/h,-C26/h,-C36/h,C26/2-C46/h,-C16/2-C56/h,-C66/h,C16/h,C26/h,C36/h,C26/2+C46/h,-C16/2+C56/h,C66/h]])
        self.Ke = np.dot(self.T.T,np.dot(self.Kel,self.T))
        # Initialize the element distributed load vector
        self.Fe = np.zeros((12,1),dtype=float)
        # Initialize the Geometric Stiffness Matrix
        kgtmp = np.zeros((12,12),dtype=float)
        kgtmp[0,0] = kgtmp[1,1] = kgtmp[6,6] = kgtmp[7,7] = 1./h
        kgtmp[0,6] = kgtmp[1,7] = kgtmp[6,0] = kgtmp[7,1] = -1./h
        self.Kegl = kgtmp
        self.Keg = np.dot(self.T.T,np.dot(self.Kegl,self.T))
        # Initialize the mass matrix
        # Create local reference of cross-section mass matrix
        M = xsect.M
        M11 = M[0,0]
        M16 = M[0,5]
        M26 = M[1,5]
        M44 = M[3,3]
        M45 = M[3,4]
        M55 = M[4,4]
        M66 = M[5,5]
        self.Mel = np.array([[h*M11/3.,0.,0.,0.,0.,h*M16/3.,h*M11/6.,0.,0.,0.,0.,h*M16/6.],\
                            [0.,h*M11/3.,0.,0.,0.,h*M26/3.,0.,h*M11/6.,0.,0.,0.,h*M26/6.],\
                            [0.,0.,h*M11/3.,-h*M16/3.,-h*M26/3.,0.,0.,0.,h*M11/6.,-h*M16/6.,-h*M26/6.,0.],\
                            [0.,0.,-h*M16/3.,h*M44/3.,h*M45/3.,0.,0.,0.,-h*M16/6.,h*M44/6.,h*M45/6.,0.],\
                            [0.,0.,-h*M26/3.,h*M45/3.,h*M55/3.,0.,0.,0.,-h*M26/6.,h*M45/6.,h*M55/6.,0.],\
                            [h*M16/3.,h*M26/3.,0.,0.,0.,h*M66/3.,h*M16/6.,h*M26/6.,0.,0.,0.,h*M66/6.],\
                            [h*M11/6.,0.,0.,0.,0.,h*M16/6.,h*M11/3.,0.,0.,0.,0.,h*M16/6.],\
                            [0.,h*M11/6.,0.,0.,0.,h*M26/6.,0.,h*M11/3.,0.,0.,0.,h*M26/3.],\
                            [0.,0.,h*M11/6.,-h*M16/6.,-h*M26/6.,0.,0.,0.,h*M11/3.,-h*M16/3.,-h*M26/3.,0.],\
                            [0.,0.,-h*M16/6.,h*M44/6.,h*M45/6.,0.,0.,0.,-h*M16/3.,h*M44/3.,h*M45/3.,0.],\
                            [0.,0.,-h*M26/6.,h*M45/6.,h*M55/6.,0.,0.,0.,-h*M26/3.,h*M45/3.,h*M55/3.,0.],\
                            [h*M16/6.,h*M26/6.,0.,0.,0.,h*M66/6.,h*M16/3.,h*M26/3.,0.,0.,0.,h*M66/3.]])
        self.Me = np.dot(self.T.T,np.dot(self.Mel,self.T))
    def applyDistributedLoad(self,fx):
        """Applies distributed load to the element.
        
        Intended primarily as a private method but left public, this method,
        applies a distributed load to the finite element. Due to the nature of
        the timoshenko beam, you cannot apply a distributed moment, however you
        can apply distributed forces.
        
        :Args:
        
        - `fx (1x6 np.array[float])`: The constant distributed load applied
            over the length of the beam.
            
        :Returns:
        
        - None
            
        """
        h = self.h
        self.Fe = np.reshape(np.array([h*fx[0]/2,h*fx[1]/2,\
                            h*fx[2]/2,h*fx[3]/2,h*fx[4]/2,h*fx[5]/2,\
                            h*fx[0]/2,h*fx[1]/2,h*fx[2]/2,h*fx[3]/2,h*fx[4]/2,\
                            h*fx[5]/2]),(12,1))
    def plotRigidBeam(self,**kwargs):
        """Plots the rigid beam in 3D space.
        
        This method plots the beam finite element in 3D space. It is not
        typically called by the beam object but by a SuperBeam object or
        even a WingSection object.
        
        :Args:
        
        - `environment (str)`: Determines what environment is to be used to
            plot the beam in 3D space. Currently only mayavi is supported.
        - `figName (str)`: The name of the figure in which the beam will apear.
        - `clr (1x3 touple(float))`: This touple contains three floats running
            from 0 to 1 in order to generate a color mayavi can plot.
                
        :Returns:
        
        - `(fig)`: The mayavi figure of the beam.
        
        """
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
        # Initialize the name of the figure
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Initialize the figure for plotting
        mlab.figure(figure=figName)
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('clr',(np.random.rand(),np.random.rand(),np.random.rand()))
        # Determine the rigid coordiates of the beam
        x1 = self.n1.x
        x2 = self.n2.x
        # Determine the tube radius:
        tube_radius = np.linalg.norm([x2-x1])/4
        # Create arrays of the coordinates for mayavi to plot
        x = np.array([x1[0],x2[0]])
        y = np.array([x1[1],x2[1]])
        z = np.array([x1[2],x2[2]])
        # Plot the beam
        if environment=='mayavi':
            mlab.plot3d(x,y,z,color=clr,tube_radius=tube_radius)
    def saveNodalDispl(self,U1,U2,**kwargs):
        """Saves applied displacements and rotations solutions if the beam.
        
        Intended primarily as a private method but left public, this method,
        save the solutions of the displacements and rotations of the beam in
        the U1 and U2 dictionary attributes. This method also calculates the
        internal forces and moments experienced by the beam under the U1 and U2
        displacements.
        
        :Args:
        
        - `U1 (MxN np.array[float])`: If N=1, this are the displacements and
            rotations of an analysis at the first node. Otherwise, this
            corresponds to the eigenvector displacements and rotations
            at the first node.
        - `U2 (MxN np.array[float])`: If N=1, this are the displacements and
            rotations of an analysis at the second node. Otherwise, this
            corresponds to the eigenvector displacements and rotations
            at the second node.
        - `analysis_name (str)`: The string of the analysis correpsonding to
            the displacement and rotation solution vector.
                
        :Returns:
        
        - None
        
        """
        # Initialize the analysis name for the analysis set
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        # Check to see if modal displacements and rotations are being saved
        if np.size(U1,axis=1)==1:
            # Save the nodal displacements for plotting purposes:
            self.U1[analysis_name] = U1
            self.U2[analysis_name] = U2
            tmpdisp = np.vstack((U1,U2))
            # Solve and save the beam internal forces and moments
            Ktop = self.Ke[0:6,:]
            Kbot = self.Ke[6:,:]
            self.F1[analysis_name] = -np.dot(Ktop,tmpdisp)
            self.F2[analysis_name] = np.dot(Kbot,tmpdisp)
        else:
            # Save the nodal displacements for plotting purposes:
            self.Umode1[analysis_name] = U1
            self.Umode2[analysis_name] = U2
            tmpdisp = np.vstack((U1,U2))
            # Solve and save the beam internal forces and moments
            Ktop = self.Ke[0:6,:]
            Kbot = self.Ke[6:,:]
            self.Fmode1[analysis_name] = -np.dot(Ktop,tmpdisp)
            self.Fmode2[analysis_name] = np.dot(Kbot,tmpdisp)
        # Save analysis name in the analysis_names attribute
        self.analysis_names+=[analysis_name]
    def plotDisplBeam(self,**kwargs):
        """Plots the displaced beam in 3D space.
        
        This method plots the deformed beam finite element in 3D space. It is
        not typically called by the beam object but by a SuperBeam object
        or even a WingSection object.
        
        :Args:
        
        - `environment (str)`: Determines what environment is to be used to
            plot the beam in 3D space. Currently only mayavi is supported.
        - `figName (str)`: The name of the figure in which the beam will apear.
        - `clr (1x3 touple(float))`: This touple contains three floats running
            from 0 to 1 in order to generate a color mayavi can plot.
        - `displScale (float)`: The scaling factor for the deformation
            experienced by the beam.
        - `mode (int)`: Determines what mode to plot. By default the mode is 0
            implying a non-eigenvalue solution should be plotted.
                
        :Returns:
        
        - `(fig)`: The mayavi figure of the beam.
        
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        mlab.figure(figure=figName)
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('clr',tuple(np.random.rand(3)))
        # Establish Beam displacement scaling
        displScale = kwargs.pop('displScale',1)
        # Determine what mode to plot:
        mode=kwargs.pop('mode',0)
        plots = kwargs.pop('plots',[])
        x1r = self.n1.x
        x2r = self.n2.x
        # Determine the tube radius:
        if hasattr(self.xsect, 'xdim'):
            tube_radius = min([self.xsect.xdim,self.xsect.ydim])
        else:
            tube_radius = np.linalg.norm([x2r-x1r])/4
        if not (analysis_name in self.F1.keys() or self.Fmode1.keys()):
            print('Warning, the analysis name for the results you are trying'+
            'to plot does not exist. The rigid beam will instead be plotted')
            self.plotRigidBeam(environment=environment,clr=clr,figName=figName\
                ,tube_radius=tube_radius)
        else:
            if mode:
                x1disp = displScale*self.Umode1[analysis_name][:,mode-1]
                x1disp = np.reshape(x1disp,(6,1))
                x2disp = displScale*self.Umode2[analysis_name][:,mode-1]
                x2disp = np.reshape(x2disp,(6,1))
            else:
                x1disp = displScale*self.U1[analysis_name]
                x2disp = displScale*self.U2[analysis_name]
            x = np.array([x1r[0]+x1disp[0],x2r[0]+x2disp[0]])
            y = np.array([x1r[1]+x1disp[1],x2r[1]+x2disp[1]])
            z = np.array([x1r[2]+x1disp[2],x2r[2]+x2disp[2]])
            if environment=='mayavi':
                line = mlab.plot3d(x,y,z,color=clr,tube_radius=tube_radius)
                plots += [line]
    def printInternalForce(self,**kwargs):
        """Prints the internal forces and moments in the beam.
        
        For a particular analysis set, this method prints out the force and
        moment resultants at both nodes of the beam.
        
        :Args:
        
        - `analysis_name (str)`: The analysis name for which the forces are
            being surveyed.
            
        :Returns:
        
        - `(str)`: This is a print out of the internal forces and moments
            within the beam element.
            
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        F1 = self.F1[analysis_name]
        F2 = self.F2[analysis_name]
        headers = ('Fx','Fy','Fz','My','Mx','Mz')
        print('Beam %05d, type=%s' %(self.EID,self.type))
        print('Forces at first node:')
        print(tabulate(np.transpose(np.around(F1,decimals=1)),headers,tablefmt="fancy_grid"))
        print('Forces at second node:')
        print(tabulate(np.transpose(np.around(F2,decimals=1)),headers,tablefmt="fancy_grid"))
    def plotWarpedXSect(self,**kwargs):
        """Plots a warped cross-section anywhere in a beam element.
        
        Intended primarily as a private method but left public, this method
        calculates the displacements and rotations as well as the force and
        moment resultants at any location within the beam. Once calculated,
        this method calls the beam's cross-section method calcWarpEffects to
        save the warping displacements, strains and stresses, and then calls
        the plotWarped method to plot those warping displacements, strains and
        stresses.
        
        :Args:
        
        - `x (float)`: The non-dimensional loation within the beam.
        - `figName (str)`: The string name of the figure
        - `contour (str)`: The contour to be plotted on the cross-section.  By
            default the Von Mises stress is returned. Currently supported
            options include: Von Mises ('VonMis'), maximum principle stress
            ('MaxPrin'), the minimum principle stress ('MinPrin'), and the
            local cross-section stress states 'sig_xx' where the subindeces can
            go from 1-3. The keyword 'none' is also an option.
        - `contLim (1x2 array[float])`: The lower and upper limits of the
            contour scale.
        - `warpScale (float)`: The scaling factor applied to all warping
            displacements within the cross-section.
        - `displScale (float)`: The scaling factor applied to all beam
            displacements and rotations.
        - `analysis_name (str)`: The string identifier associated with the
            analysis results being plotted.
        - `mode (int)`: Determines what mode to plot. By default the mode is 0
            implying a non-eigenvalue solution should be plotted.
            
        :Returns:
        
        - None
        
        """
        x = kwargs.pop('x',0.)
        if x>1. or x<0.:
            raise ValueError('The non-dimensional position "x" within the '\
                'element must be between 0. and 1.')
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Show a contour
        contour = kwargs.pop('contour','VonMis')
        # Contour Limits
        contLim = kwargs.pop('contLim',[0.,1.])
        # Establish the warping scaling factor
        warpScale = kwargs.pop('warpScale',1)
        # Select Displacement Scale
        displScale = kwargs.pop('displScale',1)
        # Analysis set name
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        # Determine what mode to plot
        mode = kwargs.pop('mode',0)
        plots = kwargs.pop('plots',[])
        # If 'analysis_untitled' is not a results key plot rigid xsect
        if not (analysis_name in self.F1.keys() or self.Fmode1.keys()):
            x_global = self.n1.x*(1.-x)+self.n2.x*(x)
            self.xsect.plotRigid(figName=figName,beam_axis=self.xbar,x=x_global)
        else:
            if mode:
                # Determine internal force at non-dimensional location
                force = self.Fmode1[analysis_name][:,mode-1]*(1.-x)+\
                    self.Fmode2[analysis_name][:,mode-1]*(x)
                # Determine internal displacement at non-dimensional location
                disp = self.Umode1[analysis_name][:,mode-1]*(1.-x)+\
                    self.Umode2[analysis_name][:,mode-1]*(x)
            else:
                # Determine internal force at non-dimensional location
                force = self.F1[analysis_name]*(1.-x)+self.F2[analysis_name]*(x)
                # Determine internal displacement at non-dimensional location
                disp = self.U1[analysis_name]*(1.-x)+self.U2[analysis_name]*(x)
            # Rotate the force in the beam from the global frame to the local
            # frame in order to recover stress and strain
            force = np.reshape(np.dot(self.T[0:6,0:6].T,force),(6,1))
            disp = np.reshape(disp,(6,1))
            x_global = self.n1.x*(1.-x)+self.n2.x*(x)
            self.xsect.calcWarpEffects(force=np.dot(self.T[0:6,0:6].T,force))
            self.xsect.plotWarped(x=x_global,U=disp,RotMat=self.T[0:3,0:3],\
                figName=figName,contour=contour,contLim=contLim,\
                displScale=displScale,warpScale=warpScale,plots=plots)
class SuperBeam:
    """Create a superbeam object.
    
    The superbeam object is mainly to fascilitate creating a whole series of
    beam objects along  the same line.
    
    :Attributes:
    
    - `type (str)`: The object type, a 'SuperBeam'.
    - `btype (str)`: The beam element type of the elements in the superbeam.
    - `SBID (int)`: The integer identifier for the superbeam.
    - `sNID (int)`: The starting NID of the superbeam.
    - `enid (int)`: The ending NID of the superbeam.
    - `xsect (obj)`: The cross-section object referenced by the beam elements
        in the superbeam.
    - `noe (int)`: Number of elements in the beam.
    - `NIDs2EIDs (dict)`: Mapping of NIDs to beam EIDs within the superbeam
    - `x1 (1x3 np.array[float])`: The 3D coordinate of the first point on the
        superbeam.
    - `x2 (1x3 np.array[float])`: The 3D coordinate of the last point on the
        superbeam.
    - `sEID (int)`: The integer identifier for the first beam element in the
        superbeam.
    - `elems (dict)`: A dictionary of all beam elements within the superbeam.
        The keys are the EIDs and the values are the corresponding beam
        elements.
    - `xbar (1x3 np.array[float])`: The vector pointing along the axis of the
        superbeam.
        
    :Methods:
    
    - `getBeamCoord`: Returns the 3D coordinate of a point along the superbeam.
    - `printInternalForce`: Prints all internal forces and moments at every
        node in the superbeam.
    - `writeDisplacements`: Writes all displacements and rotations in the
        superbeam to a .csv
    - `getEIDatx`: Provided a non-dimensional point along the superbeam, this
        method returns the local element EID and the non-dimensional
        coordinate within that element.
    - `printSummary`: Prints all of the elements and node IDs within the beam
        as well as the coordinates of those nodes.
        
    """
    def __init__(self,SBID,x1,x2,xsect,noe,btype='Tbeam',sNID=1,sEID=1,**kwargs):
        """Creates a superelement object.
        
        This method instantiates a superelement. What it effectively does is
        mesh a line provided the starting and ending points along that line.
        Keep in mind that for now, only beams running parallel to the z-axis
        are supported.
        
        :Args:
        
        - `x1 (1x3 np.array[float])`: The starting coordinate of the beam.
        - `x2 (1x3 np.array[float])`: The ending coordinate of the beam.
        - `xsect (obj)`: The cross-section used throught the superbeam.
        - `noe (int)`: The number of elements along the beam.
        - `SBID (int)`: The integer identifier for the superbeam.
        - `btype (str)`: The beam type to be meshed. Currently only Tbeam types
            are supported.
        - `sNID (int)`: The starting NID for the superbeam.
        - `sEID (int)`: The starting EID for the superbeam.
            
        :Returns:
        
        - None
        
        """
        chordVec = kwargs.pop('chordVec',np.array([1.,0.,0.]))
        # Initialize the object type
        self.type = 'SuperBeam'
        # Save the beam element type used within the superbeam.
        self.btype = btype
        # Save the SBID
        self.SBID = SBID
        # Check to make sure that the superbeam length is at least 1.
        if noe<1:
            raise ValueError('The beam super-element must contain at least 1 beam element.')
        # Store the starting NID
        self.sNID = sNID
        # Store the cross-section
        self.xsect = xsect
        # Store the number of elements
        self.noe = noe
        # Store the ending node ID
        self.enid = sNID+noe
        # Initialize a dictionary with EIDs as the keys and the associated NIDs
        # as the stored values.
        self.NIDs2EIDs = coll.defaultdict(list)
        # Create an empty element dictionary
        elems = {}
        # Parameterize the non-dimensional length of the beam
        t = np.linspace(0,1,noe+1)
        # Store the SuperBeam starting coordinate
        self.x1 = x1
        # Store the SuperBeam ending coordinate
        self.x2 = x2
        # Determine the 'slope' of the superbeam
        self.m = x2-x1
        # Store the starting element ID
        self.sEID = sEID
        tmpsnidb = sNID
        # Check which beam type is to be used:
        if btype == 'Tbeam':
            tmpsnide = sNID+1
            # Create all the elements in the superbeam
            for i in range(0,noe):
                x0 = self.getBeamCoord(t[i])
                xi = self.getBeamCoord(t[i+1])
                # Store the element in the superbeam elem dictionary
                elems[i+sEID] = TBeam(i+sEID,x0,xi,xsect,SBID=SBID,\
                    nid1=tmpsnidb,nid2=tmpsnide,chordVec=chordVec)
                self.NIDs2EIDs[tmpsnidb] += [i+sEID]
                self.NIDs2EIDs[tmpsnide] += [i+sEID]
                tmpsnidb = tmpsnide
                tmpsnide = tmpsnidb+1
        elif btype == 'EBbeam':
            tmpsnide = sNID+1
            for i in range(0,noe):
                x0 = self.getBeamCoord(t[i])
                xi = self.getBeamCoord(t[i+1])
                elems[i+sEID] = EBBeam(x0,xi,xsect,i+sEID,SBID,nid1=tmpsnidb,nid2=tmpsnide)
                self.NIDs2EIDs[tmpsnidb] += [i+sEID]
                self.NIDs2EIDs[tmpsnide] += [i+sEID]
                tmpsnidb = tmpsnide
                tmpsnide = tmpsnidb+1
        else:
            raise TypeError('You have entered an invalid beam type.')
        self.elems = elems
        # Save the unit vector pointing along the length of the beam
        self.xbar = elems[sEID].xbar
        self.RotMat = elems[sEID].T[0:3,0:3]
        nodes = {}
        for i in range(0,noe+1):
            x0 = self.getBeamCoord(t[i])
            nodes[sNID+i] = Node(sNID+i,x0)
        self.nodes = nodes
    def getBeamCoord(self,x_nd):
        """Determine the global coordinate along superbeam.
        
        Provided the non-dimensional coordinate along the beam, this method
        returns the global coordinate at that point.
        
        :Args:
        
        - `x_nd (float)`: The non-dimensional coordinate along the beam. Note
            that x_nd must be between zero and one.
            
        :Returns:
        
        - `(1x3 np.array[float])`: The global coordinate corresponding to x_nd
        """
        # Check that x_nd is between 0 and 1
        if x_nd<0. or x_nd>1.:
            raise ValueError('The non-dimensional position along the beam can'\
                'only vary between 0 and 1')
        return self.x1+x_nd*self.m
    def printInternalForce(self,**kwargs):
        """Prints the internal forces and moments in the superbeam.
        
        For every node within the superbeam, this method will print out the
        internal forces and moments at those nodes.
        
        :Args:
        
        - `analysis_name (str)`: The name of the analysis for which the forces
            and moments are being surveyed.
        
        :Returns:
        
        - `(str)`: Printed output expressing all forces and moments.
        
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        for EID, elem in self.elems.iteritems():
            elem.printInternalForce(analysis_name=analysis_name)
    def writeDisplacements(self,**kwargs):
        """Write internal displacements and rotations to file.
        
        For every node within the superbeam, this method will tabulate all of
        the displacements and rotations and then write them to a file.
        
        :Args:
        
        - `fileName (str)`: The name of the file where the data will be written.
        - `analysis_name (str)`: The name of the analysis for which the
            displacements and rotations are being surveyed.
        
        :Returns:
        
        - `fileName (file)`: This method doesn't actually return a file, rather
            it writes the data to a file named "fileName" and saves it to the
            working directory.
            
        """
        # Load default value for file name
        fileName = kwargs.pop('fileName','displacements.csv')
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        Return = kwargs.pop('Return',False)
        NID = np.zeros((len(self.elems)+1,1));
        nodeX = np.zeros((len(self.elems)+1,3));
        nodeDisp = np.zeros((len(self.elems)+1,6));
        i = 0
        NIDs = []
        for EID, elem in self.elems.iteritems():
            if not elem.n1.NID in NIDs:
                NIDs+=[elem.n1.NID]
                NID[i,0] = elem.n1.NID
                nodeX[i,:] = elem.n1.x
                nodeDisp[i,:] = elem.U1[analysis_name].T
                i+=1
            if not elem.n2.NID in NIDs:
                NIDs+=[elem.n2.NID]
                NID[i,0] = elem.n2.NID
                nodeX[i,:] = elem.n2.x
                nodeDisp[i,:] = elem.U2[analysis_name].T
                i+=1
        writeData = np.hstack((NID,nodeX,nodeDisp))
        if Return:
            return writeData
        else:
            np.savetxt(fileName,writeData,delimiter=',')
    def writeForcesMoments(self,**kwargs):
        """Write internal force and moments to file.
        
        For every node within the superbeam, this method will tabulate all of
        the forces and moments and then write them to a file.
        
        :Args:
        
        - `fileName (str)`: The name of the file where the data will be written.
        - `analysis_name (str)`: The name of the analysis for which the
            forces and moments are being surveyed.
        
        :Returns:
        
        - `fileName (file)`: This method doesn't actually return a file, rather
            it writes the data to a file named "fileName" and saves it to the
            working directory.
            
        """
        fileName = kwargs.pop('fileName','forcesMoments.csv')
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        Return = kwargs.pop('Return',False)
        NID = np.zeros((len(self.elems)+1,1));
        nodeX = np.zeros((len(self.elems)+1,3));
        nodeForce = np.zeros((len(self.elems)+1,6));
        i = 0
        NIDs = []
        for EID, elem in self.elems.iteritems():
            if not elem.n1.NID in NIDs:
                NIDs+=[elem.n1.NID]
                NID[i,0] = elem.n1.NID
                nodeX[i,:] = elem.n1.x
                nodeForce[i,:] = elem.F1[analysis_name].T
                i+=1
            if not elem.n2.NID in NIDs:
                NIDs+=[elem.n2.NID]
                NID[i,0] = elem.n2.NID
                nodeX[i,:] = elem.n2.x
                nodeForce[i,:] = elem.F2[analysis_name].T
                i+=1
        writeData = np.hstack((NID,nodeX,nodeForce))
        if Return:
            return writeData
        else:
            np.savetxt(fileName,writeData,delimiter=',')
    def getEIDatx(self,x):
        """Returns the beam EID at a non-dimensional x-location in the superbeam.
        
        Provided the non-dimensional coordinate along the beam, this method
        returns the global beam element EID, as well as the local non-
        dimensional coordinate within the specific beam element.
        
        :Args:
        
        - `x (float)`: The non-dimensional coordinate within the super-beam
        
        :Returns:
        
        - `EID (int)`: The EID of the element containing the non-dimensional
            coordinate provided.
        - `local_x_nd (float)`: The non-dimensional coordinate within the beam
            element associated with the provided non-dimensional coordinate
            within the beam.
            
        """
        '''n = len(self.elems)
        local_x_nd = 1.
        EID = max(self.elems.keys())
        for i in range(0,n):
            if x<=(float(i)/float(n)):
                EID = self.sEID+i
                local_x_nd = 1+i-n*x
                break'''
        totalLen = np.linalg.norm(self.x2-self.x1)
        xDim = x*totalLen
        for locEID, elem in self.elems.iteritems():
            localElemDim = np.linalg.norm(np.array(np.array(elem.n2.x)-self.x1))
            if xDim<=localElemDim:
                EID = locEID
                local_x_nd = (xDim-(localElemDim-elem.h))/elem.h
                break
        return EID, local_x_nd
    def printSummary(self,decimals=8,**kwargs):
        """Prints out characteristic information about the super beam.
        
        This method by default prints out the EID, XID, SBID and the NIDs along
        with the nodes associated coordinates. Upon request, it can also print
        out the beam element stiffness, geometric stiffness, mass matricies and
        distributed force vector.
        
        :Args:
        
        - `nodeCoord (bool)`: A boolean to determine if the node coordinate
            information should also be printed.
        - `Ke (bool)`: A boolean to determine if the element stiffness matrix
            should be printed.
        - `Keg (bool)`: A boolean to determine if the element gemoetric
            stiffness matrix should be printed.
        - `Me (bool)`: A boolean to determine if the element mass matrix
            should be printed.
        - `Fe (bool)`: A boolean to determine if the element distributed force
            and moment vector should be printed.
            
        :Returns:
        
        - `(str)`: Printed summary of the requested attributes.
        
        """
        # Print the associated xsect ID
        XID = kwargs.pop('XID',False)
        # Print the number of beam elements in the superbeam
        numElements = kwargs.pop('numElements',False)
        # Determine if node coordinates should also be printed
        nodeCoord = kwargs.pop('nodeCoord',True)
        # Print the stiffness matrix
        Ke = kwargs.pop('Ke',False)
        # Print the geometric stiffness matrix
        Keg = kwargs.pop('Keg',False)
        # Print the mass matrix
        Me = kwargs.pop('Me',False)
        # Print the distributed force vector
        Fe = kwargs.pop('Fe',False)
        # Print the element summaries
        
        # Print the SBID
        print('Superbeam: %d' %(self.SBID))
        if XID:
            print('Cross-section: %d' %(self.XID))
        if numElements:
            print('There are %d elements in this super-beam.' %(len(self.elems)))
        for EID, elem in self.elems.iteritems():
            elem.printSummary(nodeCoord=nodeCoord,Ke=Ke,Keg=Keg,Me=Me,Fe=Fe)
            
class WingSection:
    """Creates a wing section object.
    
    This class instantiates a wing section object which is intended to
    represent the section of a wing enclosed by two ribs. This allows primarily
    for two different things: it allows the user to vary the cross-section
    design of the wing by enabling different designs in each wing section, as
    well as enabling the user to estimate the static stability of the laminates
    that make up the wing-section design.
    
    :Attributes:
    
    - `Airfoils (Array[obj])`: This array contains all of the airfoils used
        over the wing section. This attribute exists primarily to fascilitate
        the meshing process and is subject to change.
    - `XSects (Array[obj])`: This array contains all of the cross-section
        objects used in the wing section. If the cross-section is constant
        along the length of the wing section, this array length is 1.
    - `SuperBeams (Array[obj])`: This array contains all of the superbeam
        objects used in the wing section. If the cross-section is constant
        along the length of the wing section, this array length is 1.
    - `xdim (1x2 Array[float])`: This array contains the non-dimensional
        starting and ending points of the wing section spar. They are
        non-dimensionalized by the chord length.
    - `Laminates (Array[obj])`: This array contains the laminate objects used
        by the cross-sections in the wing section.
    - `x1 (1x3 np.array[float])`: The starting coordinate of the wing section.
    - `x2 (1x3 np.array[float])`: The ending coordinate of the wing section.
    - `XIDs (Array[int])`: This array containts the integer cross-section IDs
    
    :Methods:
    
    - `plotRigid`: This method plots the rigid wing section in 3D space.
    - `plotDispl`: Provided an analysis name, this method will deformed state
        of the wing section. It is also capable of plotting cross-section
        criteria, such as displacement, stress, strain, or failure criteria.
        
    
    
    .. Warning:: While it is possible to use multiple cross-section within the
        wing section, this capability is only to be utilized for tapering cross
        sections, not changing the cross-section type or design (such as by
        changing the laminates used to make the cross-sections). Doing so would
        invalidate the ritz method buckling solutions applied to the laminate
        objects.
    
    """
    def __init__(self,x1,x2,chord,name,x0_spar,xf_spar,laminates,matLib,noe,SSBID=0,SNID=0,SEID=0,**kwargs):
        """Creates a wing section object
        
        This wing section object is in some way an organizational object. It
        holds a collection of superbeam objects which in general could all use
        different cross-sections. One could for example use several super-beams
        in order to simlate a taper within a wing section descretely. These
        objects will also be used in order to determine the buckling span of
        the laminate objects held within the cross-section.
        
        :Args:
        
        - `x1 (1x3 np.array[float])`: The starting coordinate of the wing
            section.
        - `x2 (1x3 np.array[float])`: The ending coordinate of the wing
            section.
        - `chord (func)`: A function that returns the chord length along a wing
            provided the scalar length from the wing origin to the desired
            point.
        - `name (str)`: The name of the airfoil to be used to mesh the
            cross-section. This is subject to change since the meshing process
            is only a placeholder.
        - `x0_spar (float)`: The non-dimensional starting location of the cross
            section. This value is non-dimensionalized by the local chord
            length.
        - `xf_spar (float)`: The non-dimensional ending location of the cross
            section. This value is non-dimensionalized by the local chord
            length.
        - `laminates (Array[obj])`: This array contains the laminate objects to
            be used in order to mesh the cross-section.
        - `matLib (obj)`: This material library object contains all of the
            materials to be used in meshing the cross-sections used by the
            wing section.
        - `noe (float)`: The number of beam elements to be used in the wing per
            unit length.
        - `SSBID (int)`: The starting superbeam ID in the wing section.
        - `SNID (int)`: The starting node ID in the wing section.
        - `SEID (int)`: The starting element ID in the wing section.
        - `SXID (int)`: The starting cross-section ID in the wing section.
        - `numSupBeams (int)`: The number of different superbeams to be used
            in the wing section.
        - `typeXSect (str)`: The type of cross-section used by the wing
            section.
        - `meshSize (int)`: The maximum aspect ratio an element can have within
            the cross-sections used by the wing sections.
        - `ref_ax (str)`: The reference axis used by the cross-section. This is
            axis about which the loads will be applied on the wing section.
        
        .. Note:: The chord function could take the shape of: 
            chord = lambda y: (ctip-croot)*y/b_s+croot
            
        """
        self.Airfoils = []
        self.XSects = []
        self.SuperBeams = []
        self.xdim = [x0_spar,xf_spar]
        self.laminates = laminates
        self.x1 = x1
        self.x2 = x2
        self.XIDs = []
        numSupBeams = kwargs.pop('numSuperBeams',1)
        typeXSect = kwargs.pop('typeXSect','box')
        meshSize = kwargs.pop('meshSize',4)
        SXID = kwargs.pop('SXID',0)
        ref_ax = kwargs.pop('ref_ax','shearCntr')
        chordVec = kwargs.pop('chordVec',np.array([1.,0.,0.]))
        t_vec = np.linspace(0,1,numSupBeams+1)
        xs = []
        for t in t_vec:
            xs+=[x1+t*(x2-x1)]
        tmpsnid1 = SNID
        tmpsEID = SEID
        for i in range(0,numSupBeams):
            sbeam_mid = np.linalg.norm((xs[i]+xs[i+1])/2.)
            self.Airfoils += [Airfoil(chord(sbeam_mid),name=name)]
            tmpXsect = XSect(SXID+i,self.Airfoils[i],self.xdim,\
                laminates,matLib,typeXSect=typeXSect,meshSize=meshSize)
            tmpXsect.xSectionAnalysis(ref_ax=ref_ax)
            self.XSects += [tmpXsect]
            self.XIDs += [tmpXsect.XID]
            sbeam_len = np.linalg.norm(xs[i+1]-xs[i])
            noe = int(noe*sbeam_len)
            self.SuperBeams += [SuperBeam(SSBID+i,xs[i],xs[i+1],self.XSects[i]\
                ,noe,sNID=tmpsnid1,sEID=tmpsEID,chordVec=chordVec)]
            tmpsnid1 = max(self.SuperBeams[i].nodes.keys())
            tmpsEID = max(self.SuperBeams[i].elems.keys())+1
    def plotRigid(self,**kwargs):
        """Plots the rigid wing section object in 3D space.
        
        This method is exceptionally helpful when building up a model and
        debugging it.
        
        :Args:
        
        - `figName (str)`: The name of the plot to be generated. If one is not
            provided a semi-random name will be generated.
        - `environment (str)`: The name of the environment to be used when
            plotting. Currently only the 'mayavi' environment is supported.
        - `clr (1x3 tuple(int))`: This tuple represents the RGB values that the
            beam reference axis will be colored with.
        - `numXSects (int)`: This is the number of cross-sections that will be
            plotted and evenly distributed throughout the beam.
        
        :Returns:
        
        - `(figure)`: This method returns a 3D plot of the rigid wing section.
        
        .. Warning:: In order to limit the size of data stored in memory, the
            local cross-sectional data is not stored. As a result, for every
            additional cross-section that is plotted, the time required to plot
            will increase substantially.
        
        """
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        if environment=='mayavi':
            mlab.figure(figure=figName)
            # Plot the rigid Beam Axes:
            for sbeam in self.SuperBeams:
                for EID, elem in sbeam.elems.iteritems():
                    elem.plotRigidBeam(environment=environment,clr=clr,figName=figName)
                #nids = sbeam.nodes.keys()
                # For numXSects nodes evenly spaced in the beam
                x_nd = np.linspace(0,1,numXSects)
                RotMat = sbeam.RotMat
                for i in range(0,numXSects):
                    # Determine the rigid location of the node with NID i
                    xtmp = sbeam.getBeamCoord(x_nd[i])
                    # The below lines are for loaded/displaced beams:
                    sbeam.xsect.plotRigid(figName=figName,RotMat=RotMat,x=xtmp)
    def plotDispl(self,**kwargs):
        """Plots the deformed wing section object in 3D space.
        
        Provided an analysis name, this method will plot the results from the
        corresponding analysis including beam/cross-section deformation, and
        stress, strain, or failure criteria within the sampled cross-sections.
        
        :Args:
        
        - `figName (str)`: The name of the plot to be generated. If one is not
            provided a semi-random name will be generated.
        - `environment (str)`: The name of the environment to be used when
            plotting. Currently only the 'mayavi' environment is supported.
        - `clr (1x3 tuple(int))`: This tuple represents the RGB values that the
            beam reference axis will be colored with.
        - `numXSects (int)`: This is the number of cross-sections that will be
            plotted and evenly distributed throughout the beam.
        - `contour (str)`: The contour to be plotted on the sampled cross
            sections.
        - `contLim (1x2 Array[float])`: The lower and upper limits for the
            contour color plot.
        - `warpScale (float)`: The visual multiplication factor to be applied
            to the cross-sectional warping displacement.
        - `displScale (float)`: The visual multiplication factor to be applied
            to the beam displacements and rotations.
        - `analysis_name (str)`: The analysis name corresponding to the results
            to pe visualized.
        - `mode (int)`: For modal analysis, this corresponds to the mode-shape
            which is desired to be plotted.
            
        :Returns:
        
        - `(figure)`: This method returns a 3D plot of the rigid wing section.
            
        .. Warning:: In order to limit the size of data stored in memory, the
            local cross-sectional data is not stored. As a result, for every
            additional cross-section that is plotted, the time required to plot
            will increase substantially.
            
        """
        figName = kwargs.pop('figName','Figure'+str(int(np.random.rand()*100)))
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        # Show a contour
        contour = kwargs.pop('contour','VonMis')
        # Contour Limits
        contLim = kwargs.pop('contLim',[0.,1.])
        # Establish the warping scaling factor
        warpScale = kwargs.pop('warpScale',1)
        # Select Displacement Scale
        displScale = kwargs.pop('displScale',1)
        # Analysis set name
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        # Determine what to plot
        mode = kwargs.pop('mode',0)
        plots = kwargs.pop('plots',[])
        if environment=='mayavi':
            mlab.figure(figure=figName)
            # Plot the rigid Beam Axes:
            for sbeam in self.SuperBeams:
                for EID, elem in sbeam.elems.iteritems():
                    elem.plotDisplBeam(environment=environment,clr=clr,figName=figName,\
                        displScale=displScale,analysis_name=analysis_name,mode=mode,\
                        plots=plots)
                x_nd = np.linspace(0,1,numXSects)
                # For numXSects nodes evenly spaced in the beam
                for i in range(0,numXSects):
                    tmpEID,tmpx = sbeam.getEIDatx(x_nd[i])
                    tmpElem = sbeam.elems[tmpEID]
                    tmpElem.plotWarpedXSect(x=tmpx,figName=figName,contLim=contLim,\
                        contour=contour,warpScale=warpScale,displScale=displScale,\
                        analysis_name=analysis_name,mode=mode,plots=plots)
                        # Test