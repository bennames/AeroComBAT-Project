# FEM.py
# Author: Ben Names
"""
This module contains a basic environment for conducting finite element analysis.

The primary purpose of this library is to fascilitate the creation of a FEM
within the AeroComBAT package.

:SUMARRY OF THE CLASSES:
- `Model`: The Model class has two main purposes. The first is that it is meant
    to serve as an organizational class. Once an aircraft part has been loaded
    into the model by using the addAircraftPart() method, the aircraft part
    can be loaded and constrained by the user. Once all parts have been loaded
    into the model and all loads and constraints have been applied, the user
    can choose to execute the plotRigidModel() method to visualize the model
    and make sure it accurately represents their problem. If the model appears
    as it should, the user can elect to run a static, buckling, normal mode,
    static aeroelastic, or dynamic flutter analysis.
    
.. Note:: Currently the only avaliable part in the AeroComBAT package are wing
    parts, however this is likely to change as parts such as masses, fuselages
    and other types of aircraft parts are added.
    
"""
__docformat__ = 'restructuredtext'
# =============================================================================
# IMPORT ANACONDA ASSOCIATED MODULES
# =============================================================================
import numpy as np
import scipy as sci
from tabulate import tabulate
import mayavi.mlab as mlab
from Aerodynamics import K
from Aerodynamics import calcAIC as jAIC

# =============================================================================
# DEFINE AeroComBAT FEM CLASS
# =============================================================================
class Model:
    """Creates a Model which is used to organize and analyze FEM.
    
    The primary used of Model objects are to organize FEM's and analyze them.
    The Model object doesn't create any finite elements. Instead, it loads
    aircraft parts which contain various types of finite element structural
    models as well as aerodynamic models. The type of model will depend on the
    type of aircraft part added. Once all of the models are created and added
    to the model object, the model object will serve as the analysis primary
    interface used to manipulate the generated model.
    
    :Attributes:
    - `Kg (DOFxDOF np.array[float])`: This is the global stiffness matrix.
    - `Kgr ((DOF-CON)x(DOF-CON) np.array[float])`: This is the global reduced
        stiffness matrix. In other words, the global stiffness matrix with the
        rows and columns corresponding to the constraints (CON) removed.
    - `Fg (DOFx1 np.array[float])`: The global force vector.
    - `Fgr ((DOF-CON)x1 np.array[float])`: The global reduced force vector. In
        other words, the global force vector with the rows corresponding to the
        constraints (CON) removed.
    - `Mg (DOFxDOF np.array[float])`: The global mass matrix.
    - `Mgr ((DOF-CON)x(DOF-CON) np.array[float])`: The global reduced mass
        matrix. In other words, the global mass matrix with the rows and
        columns corresponding to the constraints (CON) removed.
    - `Qg (DOFx1 np.array[float])`: The global force boundary condition vector.
        This is where all of the nodal loads are stored before the system is
        assembled.
    - `nids (Array[int])`: This array contains all of the node IDs used within
        the model.
    - `nodeDict (dict[NID,node])`: This dictionary is a mapping of the node IDs
        used within the model to the corresponding node objects.
    - `elems (Array[obj])`: This array contains all of the element objects used
        in the model.
    - `const (dict[NID,Array[DOF])`: This dictionary is a mapping of the node
        IDs constrained and the corresponding degrees of freedom that are
        constrained.
    - `parts (Array[int])`: This array contains all of the part IDs
        corresponding to the parts that have been added to the model.
    
    
    :Methods:
    - `addElements`: A method to add individual elements to the model.
    - `addAircraftParts`: A method to add an Aircraft part to the model. This
        is a much more effective method than addElements as when a part is
        added, the model can utilize all of the organizational and post
        processing methods built into the part.
    - `resetPointLoads`: A convenient way to reset all of the nodal loads in
        the model to zero.
    - `resetResults`: A convenient way to clear the results in all of the
        elements from a previous analysis. This method is subject to change as
        the way in which results are stored is likely to change.
    - `applyLoads`: A method to apply nodal loads as well as distributed loads
        to a range of elements, all of the elements in a part, or all of the
        elements in the model.
    - `applyConstraints`: A method to apply nodal constraints to the model.
    - `staticAnalysis`: A method which conducts a linear static analysis.
    - `normalModesAnalysis`: A method which conducts a normal modes analysis on
        the model.
    - `plotRigidModel`: A method to plot and visualize the model.
    - `plotDeformedModel`: A method to plot and visualize the results from an
        analysis on the model.
    
    .. Note:: When constraining nodes, only 0 displacement and rotation
        constraints are currently supported.
    
    """
    def __init__(self):
        """Instantiates the Model object.
        
        Since this method requires no inputs, the __init__() method is
        just a means to initialize attributes to be used later.
        
        :Args:
        - None
        
        
        """
        # Global Stiffness Matrix
        self.Kg = None
        # Reduced Global Stiffness Matrix
        self.Kgr = None
        # Global Force Vector
        self.Fg = None
        # Global Reduced Force Matricies
        self.Fgr = None
        # Global Mass Matrix
        self.Mg = None
        # Global Reduced Mass Matrix
        self.Mgr = None
        # Force Boundary Conditions
        self.Qg = None
        # List of node ID's contained in the model
        self.nids = []
        # Dictionary Mapping Node ID's to the global stiffness matrix
        self.nodeDict = {}
        # Numpy Object Array Containing all of the Elements in the Global System
        self.elems = []
        # The EIDs of the elements in the model.
        self.EIDs = []
        # Dictionary Mapping Node ID's to restricted degrees of Freedom
        self.const = {}
        # Array of Displacements
        self.u = None
        # Analysis ID
        self.AID=0
        #TODO: Link AID's to analysis_names
        # Parts
        self.parts = {}
        self.Loads = {}
        self.aeroBox = {}
    def addElements(self, elemarray):
        """A method to add elements to the model.
        
        Provided an array of elements, this method can add those elements to
        the model for analysis. This is a rather rudementary method as the post
        processing methods utilized by the parts are not at the users disposal
        for the elements added to the model in this way.
        
        :Args:
        - `elemarray (Array[obj])`: Adds all of the elements in the array to
            the model.
        
        :Returns:
        - None
        
        .. Note:: Currently supported elements include: SuperBeam, Tbeam.
        
        """
        # For all of the terms in the array elemarray
        for i in range(0,len(elemarray)):
            # If the ith term is a superbeam
            if elemarray[i].type=='SuperBeam':
                # Get all of the elements in the superbeam
                SBeam = elemarray[i]
                for EID, elem in SBeam.elems.iteritems():
                    if EID in self.EIDs:
                        print('Element %d not added to the model as the EID'\
                            'corresponds with an element already listed as'\
                            'already being added to the model.')
                    else:
                        self.EIDs += [EID]
                        self.elems += [elem]
                        self.nids += [elem.n1.NID,elem.n2.NID]
            elif elemarray[i].type=='Tbeam':
                TBeam = elemarray[i]
                if TBeam.EID in self.EIDs:
                    print('Element %d not added to the model as the EID'\
                        'corresponds with an element already listed as'\
                        'already being added to the model.')
                else:
                    self.EIDs += TBeam.EID
                    self.elems += [TBeam]
                    self.nids += [TBeam.n1.NID,TBeam.n2.NID]
            elif elemarray[i].type=='EBbeam':
                EBBeam = elemarray[i]
                self.elems += [EBBeam]
                self.nids += [EBBeam.n1.NID,EBBeam.n2.NID]
            else:
                raise TypeError('You have provided an object which is not a supported beam type.')
        # Remove redundant nodes
        self.nids = np.unique(self.nids)
        # Create a mapping of global nodes to matrix rows
        for i in range(0,len(self.nids)):
            self.nodeDict[self.nids[i]] = i
        # Generate an empty point load vector
        self.Qg = np.zeros(6*len(self.nids))
    def addAircraftParts(self,parts):
        """A method to add an array of aircraft parts to the model.
        
        This method is a robust version of addElements. provided an array of
        part objects, this method will add the parts to the model. This
        includes adding all of the elements and nodes to the model, as well as
        a few other pieces of information.
        
        :Args:
        - `parts (Array[obj])`: An array of part objects.
        
        :Returns:
        - None
        """
        for part in parts:
            if part.PID in self.parts:
                raise ValueError('The part ID %d is already associated with a'\
                    'part which has been added to the model.' %(part.PID))
            if part.type=='wing':
                # Add structural elements to the model
                for wingSect in part.wingSects:
                    self.addElements(wingSect.SuperBeams)
                # Add lifting surfaces to the model
                if len(part.liftingSurfaces)>0:
                    for SID, surface in part.liftingSurfaces.iteritems():
                        for PANID, panel in surface.CQUADAs.iteritems():
                            if PANID in self.aeroBox.keys():
                                return ValueError('You cannot add aero-panel %d'+
                                'CQUADA to the model if the same PANID is'+
                                'already used in the model. Consider'+
                                'renumbering your aero panel IDs' %(PANID))
                            self.aeroBox[PANID]=panel
                    # Initialize an array of PANIDs
                    PANIDs = self.aeroBox.keys()
                    # Initialize the number of panels
                    numPan = len(PANIDs)
                    Area = np.zeros((numPan,numPan))
                    # For all the recieving panels
                    Wd = np.zeros((numPan,len(self.nids)*6),dtype=complex)
                    for i in range(0,numPan):
                        recievingBox = self.aeroBox[PANIDs[i]]
                        Area[i,i] = recievingBox.Area
                        for NID, factor in recievingBox.DOF.iteritems():
                            col = self.nodeDict[NID]
                            Wd[i,col*6+2] = -1j*factor
                            Wd[i,col*6+4] = (1.+1j*recievingBox.xarm)*factor
                    self.AeroArea = Area
                    
                    self.Wd = Wd
                    self.Bd = np.dot(np.imag(Wd.T),self.AeroArea)
            self.parts[part.PID] = part
    def resetPointLoads(self):
        """A method to reset the point loads applied to the model.

        This is a good method to reset the nodal loads applied to a model. This
        method will be useful when attempting to apply a series different
        analysis.
        
        :Args:
        - None
        
        :Returns:
        - None
        
        """
        #self.Qg = np.zeros(6*len(self.nids))
        self.Loads = {}
    def resetResults(self):
        """A method to reset the results in a model.

        This is a good method to reset the results in the model from a given
        analysis. This method will be useful when attempting to apply a series
        different analysis.
        
        :Args:
        - None
        
        :Returns:
        - None
        
        """
        for part in self.parts:
            if part.type=='wing':
                for wingSect in self.wingSects:
                    for sbeam in wingSect.SuperBeams:
                        sbeam.xsect.resetResults()
    def assembleGlobalModel(self,analysisType,LID=-1,static4BuckName = 'analysis_untitled'):
        """Assembles the global model.
        
        This method adds all of the element matricies (mass, stiffness, etc.)
        to the global matrices, as well as generates the reduced matricies by
        applying constraints.
        
        :Args:
        - `static4BuckName (str)`: The analysis name of the static analysis
            should a corresponding linear buckling analysis be run.
            
        :Returns:
        - None
        """
        # For a Linear Static Analysis
        if analysisType==1:
            tmpLoad = None
            if LID in self.Loads.keys():
                tmpLoad = self.Loads[LID]
            else:
                raise ValueError('You selected a load ID that doesnt exist.')
            # Determine the degrees of freedom in the model
            DOF = 6*len(self.nids)
            # Initialize the global stiffness matrix
            Kg = np.zeros((DOF,DOF),dtype=float)
            # Initialize the global force vector
            Fg = np.zeros((DOF,1),dtype=float)
            # For all of the elements in the elems array
            for elem in self.elems:
                # Apply the distributed load to the element
                if elem.EID in tmpLoad.distributedLoads.keys():
                    elem.applyDistributedLoad(tmpLoad.distributedLoads[elem.EID])
                # Determine the node ID's associated with the elem
                nodes = [elem.n1.NID,elem.n2.NID]
                # For both NID's
                for i in range(0,len(nodes)):
                    # The row in the global matrix (an integer correspoinding to
                    # the NID)
                    row = self.nodeDict[nodes[i]]
                    # Add the elem force vector to the global matrix
                    Fg[6*row:6*row+6,:] = Fg[6*row:6*row+6,:] +\
                        elem.Fe[6*i:6*i+6,:]
                    for j in range(0,len(nodes)):
                        # Determine the column range for the NID
                        col = self.nodeDict[nodes[j]]
                        # Add the elem stiffness matrix portion to the global
                        # stiffness matrix
                        Kg[6*row:6*row+6,6*col:6*col+6] = Kg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Ke[6*i:6*i+6,6*j:6*j+6]
            # Apply the point loads to the model
            for NID in tmpLoad.pointLoads.keys():
                # The row in the global matrix (an integer correspoinding to
                # the NID)
                row = self.nodeDict[NID]
                Fg[6*row:6*row+6,:]=Fg[6*row:6*row+6,:]\
                    +np.reshape(tmpLoad.pointLoads[NID],(6,1))
            # Save the global stiffness matrix
            self.Kg = Kg
            # Save the global force vector
            self.Fg = Fg
            # Determine the list of NIDs to be contrained
            cnds = sorted(list(self.const.keys()))
            # Initialize the number of equations to be removed from the system
            deleqs = 0
            # For the number of constrained NIDs
            for i in range(0,len(self.const)):
                # The row range to be removed associated with the NID
                row = self.nodeDict[cnds[i]]
                # Determine which DOF are to be removed
                tmpcst = self.const[cnds[i]]
                # For all of the degrees of freedom to be removed
                for j in range(0,len(tmpcst)):
                    # Remove the row associated with the jth DOF for the ith NID
                    Fg = np.delete(Fg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    # Incremend the number of deleted equations
                    deleqs += 1
            # Save the reduced global force vector
            self.Fgr = Fg
            # Save the reduced global stiffness matrix
            self.Kgr = Kg
            
        # For a linear buckling analysis
        if analysisType==2:
            # Determine the degrees of freedom in the model
            DOF = 6*len(self.nids)
            # Initialize the geometric global stiffness matrix
            Kgg = np.zeros((DOF,DOF),dtype=float)
            # Initialize the global mass matrix
            for elem in self.elems:
                # Determine the node ID's associated with the elem
                nodes = [elem.n1.NID,elem.n2.NID]
                # For both NID's
                for i in range(0,len(nodes)):
                    # The row in the global matrix (an integer correspoinding to
                    # the NID)
                    row = self.nodeDict[nodes[i]]
                    for j in range(0,len(nodes)):
                        # Determine the column range for the NID
                        col = self.nodeDict[nodes[j]]
                        # Determine the axial force in the beam
                        if static4BuckName in elem.F1.keys():
                            Ploc = elem.F1[static4BuckName][2]
                        else:
                            Ploc=0.
                        # Add the elem geometric stiffness matrix portion to the
                        # global stiffness matrix
                        Kgg[6*row:6*row+6,6*col:6*col+6] = Kgg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Keg[6*i:6*i+6,6*j:6*j+6]*Ploc
            # Save the global geometric stiffness matrix
            self.Kgg = Kgg
            # Determine the list of NIDs to be contrained
            cnds = sorted(list(self.const.keys()))
            # Initialize the number of equations to be removed from the system
            deleqs = 0
            # For the number of constrained NIDs
            for i in range(0,len(self.const)):
                # The row range to be removed associated with the NID
                row = self.nodeDict[cnds[i]]
                # Determine which DOF are to be removed
                tmpcst = self.const[cnds[i]]
                # For all of the degrees of freedom to be removed
                for j in range(0,len(tmpcst)):
                    # Remove the row associated with the jth DOF for the ith NID
                    Kgg = np.delete(Kgg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kgg = np.delete(Kgg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    # Increment the number of deleted equations
                    deleqs += 1
            # Save the reduced global geometric stiffness matrix
            self.Kggr = Kgg
            
        # For a Normal Modes Analysis
        if analysisType==3:
            # Determine the degrees of freedom in the model
            DOF = 6*len(self.nids)
            # Initialize the global stiffness matrix
            Kg = np.zeros((DOF,DOF),dtype=float)
            Mg = np.zeros((DOF,DOF),dtype=float)
            # For all of the elements in the elems array
            for elem in self.elems:
                # Determine the node ID's associated with the elem
                nodes = [elem.n1.NID,elem.n2.NID]
                # For both NID's
                for i in range(0,len(nodes)):
                    # The row in the global matrix (an integer correspoinding to
                    # the NID)
                    row = self.nodeDict[nodes[i]]
                    for j in range(0,len(nodes)):
                        # Determine the column range for the NID
                        col = self.nodeDict[nodes[j]]
                        # Add the elem stiffness matrix portion to the global
                        # stiffness matrix
                        Kg[6*row:6*row+6,6*col:6*col+6] = Kg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Ke[6*i:6*i+6,6*j:6*j+6]
                        # Add the element mass matrix portion to the global mass matrix
                        Mg[6*row:6*row+6,6*col:6*col+6] = Mg[6*row:6*row+6,6*col:6*col+6]\
                                                        +elem.Me[6*i:6*i+6,6*j:6*j+6]                        
            # Save the global stiffness matrix
            self.Kg = Kg
            # Save the global mass matrix
            self.Mg = Mg
            # Determine the list of NIDs to be contrained
            cnds = sorted(list(self.const.keys()))
            # Initialize the number of equations to be removed from the system
            deleqs = 0
            # For the number of constrained NIDs
            for i in range(0,len(self.const)):
                # The row range to be removed associated with the NID
                row = self.nodeDict[cnds[i]]
                # Determine which DOF are to be removed
                tmpcst = self.const[cnds[i]]
                # For all of the degrees of freedom to be removed
                for j in range(0,len(tmpcst)):
                    # Remove the row associated with the jth DOF for the ith NID
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                    Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                    # Incremend the number of deleted equations
                    deleqs += 1
            # Save the reduced global stiffness matrix
            self.Kgr = Kg
            # Save the reduced global mass matrix
            self.Mgr = Mg
            
    def applyLoads(self,LID,**kwargs):
        """A method to apply nodal and distributed loads to the model.
        
        This method allows the user to apply nodal loads to nodes and
        distributed loads to elements within the model.
        
        :Args:
        - `f (func)`: A function which, provided the provided a length 3 numpy
            array representing a point in space, calculates the distributed
            load value at that point. See an example below:
        - `F (dict[NID,1x6 np.array[float]])`: A dictionary mapping a node ID
            to the loads to be applied at that node ID.
        - `allElems (bool)`: A boolean value used to easily load all of the
            elements which have been added to the model.
        - `PIDs (Array[int])`: An array containing part ID's, signifying that
            all elements used by that part should be loaded.
        - `eids (Array[int])`: An array containing all of the element ID's
            corresponding to all of the elements which should be loaded.
        
        :Returns:
        - None
        
        Distributed load function example:
        
        .. code-block:: python
        
           def f(x):
              vx = (1/10)*10*x[2]**2-7*x[2]-2.1
              vy = 10*x[2]**2-7*x[2]
              pz = 0
              mx = 0
              my = 0
              tz = (10*x[2]**2-7*x[2])/10+3*x[0]**2
              return np.array([vx,vy,pz,mx,my,tz])
        
        Nodal load dictionary example:
        
        .. code-block:: python
        
           F[NID] = np.array([Qx,Qy,P,Mx,My,T])
        """
        if not LID in self.Loads.keys():
            self.Loads[LID] = LoadSet(LID)
        tmpLoad = self.Loads[LID]
        # TODO: Make it so that local CSYS can be used for load applications. This
        # Should allow for translation and rotations.
        def fdefault(x):
                vx = 0.
                vy = 0.
                pz = 0.
                mx = 0.
                my = 0.
                tz = 0.
                return np.array([vx,vy,pz,mx,my,tz])
        # Get the distributed load function
        f = kwargs.pop('f',fdefault)
        # Get the descrete load dictionary
        F = kwargs.get('F')
        # Keyword to fascilitate applying distributed load to all elements
        allElems = kwargs.pop('allElems',False)
        PIDs = kwargs.pop('PIDs',[])
        eids = kwargs.pop('eids',[])
        
        if allElems:
            eids += [elem.EID for elem in self.elems]
            if not len(PIDs)==0:
                print('WARNING: You are atempting to simultaneiously load all'\
                    'elements within the model as well as all elements in'\
                    'parts:')
                print(PIDs)
        else:
            if not len(PIDs)==0:
                for PID in PIDs:
                    part = self.parts[PID]
                    if part.type=='wing':
                        for wingSect in part.wingSects:
                            for superBeam in wingSect.SuperBeams:
                                eids+=superBeam.elems.keys()
        if len(eids)==0 and not f==fdefault:
            print('WARNING: You are attempting to apply a distributed load'\
                'without listing EIDs at which to apply those loads.')
        # Remove rudundant EIDs
        eids = np.unique(eids)
        # For all of the in the model
        for elem in self.elems:
            if elem.EID in eids:
                if elem.type=='Tbeam':
                    x1 = elem.n1.x
                    x2 = elem.n2.x
                    fx = f((x1+x2)/2)
                    tmpLoad.addDistributedLoad(fx,elem.EID)
        if F==None:
            pass
        else:
            for NID in F.keys():
                tmpLoad.addPointLoad(F[NID],NID)
            '''
            # Determine what global NIDs the point loads are to be applied at
            keys = list(F.keys())
            if (self.Qg).any()==None:
                raise Exception('There are no elements in this model to apply point loads to.')
            # For the NIDS, populate the global applied force vector
            for key in keys:
                Ftmp = F[key]
                if not len(Ftmp)==6:
                    raise ValueError('When you apply a discrete force to a node'\
                        'it must be a numpy array of exactly length 3.')
                Ftmp = np.array([Ftmp[0],Ftmp[1],Ftmp[2],Ftmp[3],Ftmp[4],Ftmp[5]])
                self.Qg[6*self.nodeDict[key]:6*self.nodeDict[key]+6] = \
                self.Qg[6*self.nodeDict[key]:6*self.nodeDict[key]+6]+Ftmp'''
    def applyConstraints(self,NID,const):
        """A method for applying nodal constraints to the model.
        
        This method is the primary method for applying nodal constraints to the
        model.
        
        :Args:
        - `NID (int)`: The node ID of the node to be constrained.
        - `const (str, np.array[int])`: const can either take the form of a
            string in order to take advantage of the two most common
            constraints being 'pin' or 'fix'. If a different constraint needs
            to be applied, const could also be a numpy array listing the DOF
            (integers 1-6) to be constrained.
        
        :Returns:
        - None
        """
        # INPUTS:
        # nid - The node you want to constrain
        # const - a numpy array containing integers from 1-6 or a string description.
        # For example, to pin a beam in all three directions, the const array would look like:
        #  const = np.array([1,2,3],dtype=int) = 'pin'
        # A fully fixed node constraint would look like:
        #  const = np.array([1,2,3,4,5,6],dtype=int) = 'fix'
        const = np.unique(const)
        if len(const)>6:
            raise ValueError('Too many constraints have been applied than are possible.')
        if const=='pin':
            self.const[NID]=np.array([1,2,3],dtype=int)
        elif const=='fix':
            self.const[NID]=np.array([1,2,3,4,5,6],dtype=int)
        else:
            self.const[NID]=const
    def staticAnalysis(self,LID,**kwargs):
        """Linear static analysis.
        
        This method conducts a linear static analysis on the model. This will
        calculate all of the unknown displacements in the model, and save not
        only dispalcements, but also internal forces and moments in all of the
        beam elements.
        
        :Args:
        - `analysis_name (str)`: The string name to be associated with this
            analysis. By default, this is chosen to be 'analysis_untitled'.
        
        :Returns:
        - None
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.assembleGlobalModel(1,LID=LID)
        # Prepare the reduced stiffness matrix for efficient LU decomposition solution
        lu,piv = sci.linalg.lu_factor(self.Kgr)
        # Solve global displacements
        u = sci.linalg.lu_solve((lu,piv),self.Fgr)
        self.ur = u
        # Generate list of constraint keys
        ckeys = sorted(list(self.const.keys()))
        # For each node constrained
        for i in range(0,len(ckeys)):
            # Establish temporary node constrined
            tmpconst = self.const[ckeys[i]]
            # For each DOF contrained on the temporary node
            for j in range(0,len(tmpconst)):
                #Insert a zero for the "displacement"
                u = np.insert(u,self.nodeDict[ckeys[i]]*6+(tmpconst[j]-1),0,axis=0)
        self.u = u
        #Solve for the reaction forces in the elements
        # For all of the beam elements in the model
        for elem in self.elems:
            # If the element is a Tbeam
            if (elem.type=='Tbeam'):
                #Populate the local nodal displacements:
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                U1 = u[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6]
                U2 = u[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6]
                elem.saveNodalDispl(U1,U2,analysis_name=analysis_name)
            elif elem.type=='EBbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                elem.U1 = u[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6]
                elem.U2 = u[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6]
                #Solve for the reaction forces on the first node
                Ke12 = elem.Ke[0:6,6:12]
                #elem.F1 = np.dot(Ke12,elem.U2)+np.dot(Ke13,elem.U3)
                elem.F1 = np.dot(Ke12,elem.U1-elem.U2)
                #Solve for the reaction forces on the second node
                Ke21 = elem.Ke[6:12,0:6]
                elem.F2 = np.dot(Ke21,elem.U1-elem.U2)
    def bucklingAnalysis(self,LID,**kwargs):
        static_analysis_name = kwargs.pop('static_analysis_name','static_analysis_untitled')
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.staticAnalysis(LID,analysis_name=static_analysis_name)
        self.assembleGlobalModel(2,static4BuckName=static_analysis_name)
        Fscale,umode = sci.linalg.eig(self.Kgr,-self.Kggr)
        idx = Fscale.argsort()
        self.Fscale = np.array(Fscale[idx],dtype=float)
        umode = np.array(umode[:,idx],dtype=float)
        #Generate list of constraint keys
        ckeys = sorted(list(self.const.keys()))
        #For each node constrained
        for i in range(0,len(ckeys)):
            #Establish temporary node constrined
            tmpconst = self.const[ckeys[i]]
            #For each DOF contrained on the temporary node
            for j in range(0,len(tmpconst)):
                #Insert a zero for the constrained degrees of Freedom
                umode = np.insert(umode,self.nodeDict[ckeys[i]]*6+(tmpconst[j]-1),np.zeros((1,len(Fscale))),axis=0)
        self.umode = umode
        for elem in self.elems:
            if elem.type=='Tbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                elem.Umode1 = umode[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6,:]
                elem.Umode2 = umode[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6,:]
            elif elem.type=='EBbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                elem.Umode1 = umode[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6,:]
                elem.Umode2 = umode[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6,:]
    #TODO: Create a method to print displacements. Maybe in the NASTRAN .f06 format?
    def normalModesAnalysis(self,**kwargs):
        """Conducts normal mode analysis.
        
        This method conducts normal mode analysis on the model. This will
        calculate all of the unknown frequency eigenvalues and eigenvectors for
        the model, which can be plotted later.
        
        
        :Args:
        - `analysis_name (str)`: The string name to be associated with this
            analysis. By default, this is chosen to be 'analysis_untitled'.
        
        :Returns:
        - None
        
        .. Note:: There are internal loads that are calculated and stored
            within the model elements, however be aware that these loads are
            meaningless and are only retained as a means to display cross
            section warping.
        """
        # Create Analysis Name
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.assembleGlobalModel(3)
        eigs,umode = sci.linalg.eig(self.Kgr,self.Mgr)
        idx = eigs.argsort()
        self.freqs = np.sqrt(np.array(eigs[idx],dtype=float))/(2*np.pi)
        umode = np.array(umode[:,idx],dtype=float)
        self.umoder = umode
        #Generate list of constraint keys
        ckeys = sorted(list(self.const.keys()))
        #For each node constrained
        for i in range(0,len(ckeys)):
            #Establish temporary node constrined
            tmpconst = self.const[ckeys[i]]
            #For each DOF contrained on the temporary node
            for j in range(0,len(tmpconst)):
                #Insert a zero for the constrained degrees of Freedom
                umode = np.insert(umode,self.nodeDict[ckeys[i]]*6+(tmpconst[j]-1),np.zeros((1,len(eigs))),axis=0)
        self.umode = umode
        for elem in self.elems:
            if elem.type=='Tbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                Umode1 = umode[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6,:]
                Umode2 = umode[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6,:]
                elem.saveNodalDispl(Umode1,Umode2,analysis_name=analysis_name)
    def flutterAnalysis(self,U_vec,kr_vec,M_vec,b,rho_0,nModes,**kwargs):
        """Conducts a flutter analysis.
        
        This method calculates the flutter modes and damping provided
        velocities, reduced frequencies, Mach numbers, and the reference
        semi-chord.
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        rho_rat = kwargs.pop('rho_rat',np.ones(len(U_vec)))
        g = kwargs.pop('g',0.)
        symxz = kwargs.pop('symxz',False)
        #Generate list of constraint keys
        ckeys = sorted(list(self.const.keys()))
        # Assemble Kgr and Mgr matricies
        #self.assembleGlobalModel(3)
        self.normalModesAnalysis()
        normalModes = self.umode
        # Number of DOF for the global system
        DOF = np.size(self.Kg,axis=0)
        # Create flutter point objects
        self.flutterPoints = {}
        for i in range(0,nModes):
            self.flutterPoints[i] = FlutterPoint(i,U_vec,nModes)
        # Determine the modally reduced stiffness matrix:
        #for k in range(0,len(ckeys)):
        #    # Establish temporary node constrined
        #    tmpconst = self.const[ckeys[k]]
        #    # For each DOF contrained on the temporary node
        #    for l in range(0,len(tmpconst)):
        #        # Insert a zero for the constrained degrees of Freedom
        #        modes = np.insert(modes,self.nodeDict[ckeys[k]]*6+\
        #            (tmpconst[l]-1),np.zeros((1,len(eigs))),axis=0)
        nrmModes = self.umoder[:,0:nModes]
        Kgrm = np.dot(nrmModes.T,np.dot(self.Kgr,nrmModes))
        Mgrm = np.dot(nrmModes.T,np.dot(self.Mgr,nrmModes))
        
        self.TestEig = []
        self.TestMAC = []
        self.TestMode = []
        # For all reduced frequencies
        for kr, M in zip(kr_vec, M_vec):
            print(kr)
            #tmpModes = np.copy(normalModes)
            tmpModes = np.eye(nModes)
            self.calcAIC(M,kr,b,symxz=symxz)
            for U, rhoRat in zip(U_vec, rho_rat):
                # Calculate the Qaic multiplied by dynamic pressure
                #Qaicr = self.Qaicr*0.5*rhoRat*rho_0*U**2
                Qaicrm = np.dot(nrmModes.T,np.dot(self.Qaicr*0.5*rhoRat*rho_0*U**2,nrmModes))
                #eigs, modes = sci.linalg.eig(-np.dot(np.linalg.inv(self.Mgr),(1+1j*g)*self.Kgr-Qaicr))
                eigs, modes = sci.linalg.eig(-np.dot(np.linalg.inv(Mgrm),(1+1j*g)*Kgrm-Qaicrm))
                eigs = np.sqrt(eigs)
                        #eigs[k]=np.conj(eigs[k])
                #self.testEigs = eigs
                #raise ValueError('Test')
                #eigs, modes = sci.linalg.eig(A,B)
                # For each node constrained
                
                for k in range(0,len(eigs)):
                    if np.imag(eigs[k])<0:
                        eigs[k]=-eigs[k]
                # MAC
                #MAC = np.divide(np.dot(modes.T,np.conj(tmpModes))**2,\
                #    np.multiply(np.dot(modes.T,np.conj(modes)),np.dot(tmpModes.T,np.conj(tmpModes))))
                MAC = np.zeros((np.size(modes,axis=1),np.size(modes,axis=1)))
                for i in range(0,np.size(modes,axis=1)):
                    for j in range(0,np.size(modes,axis=1)):
                        #v0 = tmpModes[:,j]
                        #v1 = modes[:,i]
                        #eig0 = 1/eigs[j]
                        #eig1 = 1/eigs[i]
                        #num = (abs(np.dot(np.conj(v0),v1))/abs(np.conj(eig0)+eig1)+\
                        #    abs(np.dot(v0,v1))/abs(eig0+eig1))**2
                        #den = (np.dot(np.conj(v0),v0)/(2*abs(np.real(eig0)))+\
                        #    abs(np.dot(v0,v0))/(2*abs(eig0)))*\
                        #    (np.dot(np.conj(v1),v1)/(2*abs(np.real(eig1)))+\
                        #    abs(np.dot(v1,v1))/(2*abs(eig1)))
                        MAC[i,j] = abs(np.dot(np.conj(tmpModes[:,j]),modes[:,i])**2/\
                            (np.dot(tmpModes[:,j],np.conj(tmpModes[:,j]))\
                            *np.dot(modes[:,i],np.conj(modes[:,i]))))
                        #MAC[i,j] = num/den
                #MAC = np.dot(tmpModes.T,np.conj(modes))**2
                self.MAC = MAC
                idx = []
                for i in range(0,np.size(MAC,axis=1)):
                    idx += [np.argmax(MAC[:,i])]
                self.eigs = eigs
                #idx = np.imag(eigs).argsort()
                p = eigs[idx]
                self.p = p
                modes = modes[:,idx]
                self.idx = idx
                self.tmpModes = tmpModes
                tmpModes = modes
                self.modes = modes
                #p = np.sqrt(eigs[0:nModes])
                omega_root = np.imag(p)
                gamma_root = np.real(p)/omega_root
                omega_aero = kr*U/b
                #raise ValueError('Test')
                #if U==U_vec[13] and kr==kr_vec[1]:
                #    raise ValueError('test')
                for i in range(0,nModes):
                    #print(omega_root[i])
                    self.flutterPoints[i].saveSol(U,omega_aero,\
                        omega_root[i],gamma_root[i],np.real(modes[:,i]))
        for FPID, flutterPoint in self.flutterPoints.iteritems():
            flutterPoint.interpOmegaRoot()
            
    def jitflutterAnalysis(self,U_vec,kr_vec,M_vec,b,rho_0,nModes,**kwargs):
        """Conducts a flutter analysis.
        
        This method calculates the flutter modes and damping provided
        velocities, reduced frequencies, Mach numbers, and the reference
        semi-chord.
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        rho_rat = kwargs.pop('rho_rat',np.ones(len(U_vec)))
        g = kwargs.pop('g',0.)
        symxz = kwargs.pop('symxz',False)
        # Assemble Kgr and Mgr matricies
        #self.assembleGlobalModel(3)
        self.normalModesAnalysis()
        # Create flutter point objects
        self.flutterPoints = {}
        for i in range(0,nModes):
            self.flutterPoints[i] = FlutterPoint(i,U_vec,nModes)
        nrmModes = self.umode[:,0:nModes]
        Kgm = np.dot(nrmModes.T,np.dot(self.Kg,nrmModes))
        Mgm = np.dot(nrmModes.T,np.dot(self.Mg,nrmModes))
        delta_x_vec = []
        sweep_vec = []
        l_vec = []
        dihedral_vec = []
        numPanels = len(self.aeroBox)
        Xr_vec = np.zeros((numPanels,3))
        Xi_vec = np.zeros((numPanels,3))
        Xc_vec = np.zeros((numPanels,3))
        Xo_vec = np.zeros((numPanels,3))
        for PANID, panel in self.aeroBox.iteritems():
            delta_x_vec += [panel.delta_x]
            sweep_vec += [panel.sweep]
            l_vec += [panel.l]
            dihedral_vec += [panel.dihedral]
            Xr_vec[PANID,:] = panel.Xr
            Xi_vec[PANID,:] = panel.Xi
            Xc_vec[PANID,:] = panel.Xc
            Xo_vec[PANID,:] = panel.Xo
        # For all reduced frequencies
        for kr, M in zip(kr_vec, M_vec):
            print(kr)
            tmpModes = np.eye(nModes)
            D = jAIC(M,kr,b,delta_x_vec,sweep_vec,l_vec,dihedral_vec,\
                Xr_vec,Xi_vec,Xc_vec,Xo_vec,symxz=symxz)
            W = np.real(self.Wd)+1j*kr/b*np.imag(self.Wd)
            Qaic = np.dot(self.Bd,np.dot(np.linalg.inv(D),W))
            for U, rhoRat in zip(U_vec, rho_rat):
                # Calculate the Qaic multiplied by dynamic pressure
                Qaicm = np.dot(nrmModes.T,np.dot(Qaic*0.5*rhoRat*rho_0*U**2,nrmModes))
                eigs, modes = sci.linalg.eig(-np.dot(np.linalg.inv(Mgm),(1+1j*g)*Kgm-Qaicm))
                eigs = np.sqrt(eigs)
                for k in range(0,len(eigs)):
                    if np.imag(eigs[k])<0:
                        eigs[k]=-eigs[k]
                # MAC
                MAC = np.zeros((np.size(modes,axis=1),np.size(modes,axis=1)))
                for i in range(0,np.size(modes,axis=1)):
                    for j in range(0,np.size(modes,axis=1)):
                        MAC[i,j] = abs(np.dot(np.conj(tmpModes[:,j]),modes[:,i])**2/\
                            (np.dot(tmpModes[:,j],np.conj(tmpModes[:,j]))\
                            *np.dot(modes[:,i],np.conj(modes[:,i]))))
                self.MAC = MAC
                idx = []
                for i in range(0,np.size(MAC,axis=1)):
                    idx += [np.argmax(MAC[:,i])]
                self.eigs = eigs
                #idx = np.imag(eigs).argsort()
                p = eigs[idx]
                self.p = p
                modes = modes[:,idx]
                self.idx = idx
                self.tmpModes = tmpModes
                tmpModes = modes
                self.modes = modes
                #p = np.sqrt(eigs[0:nModes])
                omega_root = np.imag(p)
                gamma_root = np.real(p)/omega_root
                omega_aero = kr*U/b
                for i in range(0,nModes):
                    #print(omega_root[i])
                    self.flutterPoints[i].saveSol(U,omega_aero,\
                        omega_root[i],gamma_root[i],np.real(modes[:,i]))
        for FPID, flutterPoint in self.flutterPoints.iteritems():
            flutterPoint.interpOmegaRoot()            
            
    def plotRigidModel(self,**kwargs):
        """Plots the rigid model.
        
        This method plots the rigid model in the mayavi environement.
        
        :Args:
        - `figName (str)`: The name of the figure. This is 'Rigid Model' by
            default.
        - `clr (1x3 tuple(int))`: The color tuple or RGB values to be used for
            plotting the reference axis for all beam elements. By default this
            color is black.
        - `numXSects (int)`: The number of cross-sections desired to be plotted
            for all wing sections. The default is 2.
        
        :Returns:
        - mayavi figure
        
        """
        figName = kwargs.pop('figName','Rigid Model')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        mlab.figure(figure=figName)
        for PID, part in self.parts.iteritems():
            if part.type=='wing':
                part.plotRigidWing(figName=figName,clr=clr,numXSects=numXSects)
    def plotDeformedModel(self,**kwargs):
        """Plots the deformed model.
        
        This method plots the deformed model results for a given analysis in
        the mayavi environement.
        
        :Args:
        - `analysis_name (str)`: The string identifier of the analysis.
        - `figName (str)`: The name of the figure. This is 'Rigid Model' by
            default.
        - `clr (1x3 tuple(int))`: The color tuple or RGB values to be used for
            plotting the reference axis for all beam elements. By default this
            color is black.
        - `numXSects (int)`: The number of cross-sections desired to be plotted
            for all wing sections. The default is 2.
        - `contour (str)`: A string keyword to determine what analysis should
            be plotted.
        - `contLim (1x2 Array[float])`: An array containing the lower and upper
            contour limits.
        - `warpScale (float)`: The scaling factor used to magnify the cross
            section warping displacement factor.
        - `displScale (float)`: The scaling fator used to magnify the beam
            element displacements and rotations.
        - `mode (int)`: If the analysis name refers to a modal analysis, mode
            refers to which mode from that analysis should be plotted.
        
        :Returns:
        - mayavi figure
        
        """
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        figName = kwargs.pop('figName','Deformed Wing')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        # Show a contour
        contour = kwargs.pop('contour','VonMis')
        # Stress Limits
        contLim = kwargs.pop('contLim',[0.,1.])
        # Establish the warping scaling factor
        warpScale = kwargs.pop('warpScale',1)
        # Establish Beam displacement scaling
        displScale = kwargs.pop('displScale',1)
        # Which mode to plot. Note by default mode=0 implies not plotting an
        # eigenvalue solution.
        mode = kwargs.pop('mode',0)
        mlab.figure(figure=figName)
        for PID, part in self.parts.iteritems():
            if part.type=='wing':
                for sects in part.wingSects:
                    sects.plotDispl(figName=figName,clr=clr,numXSects=numXSects,\
                        contLim=contLim,warpScale=warpScale,displScale=displScale,\
                        contour=contour,analysis_name=analysis_name,mode=mode)
        mlab.colorbar()
        
    def calcAIC(self,M,kr,br,symxz=False):
        # Calculate the AIC matricies
        # Initialize an array of PANIDs
        PANIDs = self.aeroBox.keys()
        # Initialize the number of panels
        numPan = len(PANIDs)
        # Initialize the complex [D] matrix
        D = np.zeros((numPan,numPan),dtype=complex)
        ## Initialize the Diagonal Box Area Matrix
        #Area = np.zeros((numPan,numPan))
        # Initialize the [W] Matrix
        W = np.zeros((numPan,len(self.nids)*6),dtype=complex)
        # For all the recieving panels
        for i in range(0,numPan):
            recievingBox = self.aeroBox[PANIDs[i]]
            # For all the sending panels
            for j in range(0,numPan):
                sendingBox = self.aeroBox[PANIDs[j]]
                # Calculate average chord of sending box
                #delta_x_j = abs(sendingBox.x(0,1)-sendingBox.x(0,-1))
                delta_x_j = sendingBox.delta_x
                # Calculate sweep of sending box
                '''xtmp = sendingBox.x(1,1)-sendingBox.x(-1,1)
                ytmp = sendingBox.y(1,1)-sendingBox.y(-1,1)
                lambda_j = np.arctan(xtmp/ytmp)'''
                lambda_j = sendingBox.sweep
                # Calculate the length of the doublet line on sending box
                '''xtmp = sendingBox.x(1,0.5)-sendingBox.x(-1,0.5)
                ytmp = sendingBox.y(1,0.5)-sendingBox.y(-1,0.5)
                ztmp = sendingBox.z(1,0.5)-sendingBox.z(-1,0.5)
                l_j = np.linalg.norm([xtmp,ytmp,ztmp])'''
                l_j = sendingBox.l
                # Calculate y and z vectors for recieving box
                '''ytmp_r = recievingBox.y(1,0.5)-recievingBox.y(-1,0.5)
                ztmp_r = recievingBox.z(1,0.5)-recievingBox.z(-1,0.5)'''
                # Calculate parameters invloved in aproximate I_ij
                '''Xr = np.array([recievingBox.x(0,-.5),recievingBox.y(0,-.5),\
                    recievingBox.z(0,-.5)])
                Xi = np.array([sendingBox.x(-1,.5),sendingBox.y(-1,.5),\
                    sendingBox.z(-1,.5)])
                Xc = np.array([sendingBox.x(0,.5),sendingBox.y(0,.5),\
                    sendingBox.z(0,.5)])
                Xo = np.array([sendingBox.x(1,.5),sendingBox.y(1,.5),\
                    sendingBox.z(1,.5)])'''
                Xr = recievingBox.Xr
                Xi = sendingBox.Xi
                Xc = sendingBox.Xc
                Xo = sendingBox.Xo
                e = 0.5*l_j*np.cos(lambda_j)
                #gamma_s = np.arctan(ztmp/ytmp)
                gamma_s = sendingBox.dihedral
                gamma_r = recievingBox.dihedral
                eta_0 = (Xr[1]-Xc[1])*np.cos(gamma_s)\
                    +(Xr[2]-Xc[2])*np.sin(gamma_s)
                zeta_0 = -(Xr[1]-Xc[1])*np.sin(gamma_s)\
                    +(Xr[2]-Xc[2])*np.cos(gamma_s)
                r1 = np.linalg.norm([eta_0,zeta_0])
                # Calculate the Kernel function at the inboard, middle, and
                # outboard locations
                #if multiProcess:
                #    print('start multiprocess')
                #    Ki,Kc,Ko = K_multi(Xr,Xi,Xc,Xo,gamma_r,gamma_s,M,br,kr,r1)
                Ki = K(Xr,Xi,gamma_r,gamma_s,M,br,kr,r1)
                Kc = K(Xr,Xc,gamma_r,gamma_s,M,br,kr,r1)
                Ko = K(Xr,Xo,gamma_r,gamma_s,M,br,kr,r1)
                #print('got Ks')
                A = (Ki-2*Kc+Ko)/(2*e**2)
                B = (Ko-Ki)/(2*e)
                C = Kc
                # Determine if planar or non-planar I_ij definition should be used
                if abs(zeta_0)<1e-6:
                    I_ij = (eta_0**2*A+eta_0*B+C)*(1./(eta_0-e)-1./(eta_0+e))+\
                        (B/2+eta_0*A)*np.log(((eta_0-e)/(eta_0+e))**2)
                else:
                    I_ij = ((eta_0**2-zeta_0**2)*A+eta_0*B+C)*zeta_0**(-1)*\
                        np.arctan(2*e*abs(zeta_0)/(r1**2-e**2))+\
                        (B/2+eta_0*A)*np.log((r1**2-2*eta_0*e+e**2)/\
                        (r1**2+2*eta_0*e+e**2))+2*e*A
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
                    eta_0 = (Xr[1]-Xc[1])*np.cos(gamma_s)\
                        +(Xr[2]-Xc[2])*np.sin(gamma_s)
                    zeta_0 = -(Xr[1]-Xc[1])*np.sin(gamma_s)\
                        +(Xr[2]-Xc[2])*np.cos(gamma_s)
                    r1 = np.linalg.norm([eta_0,zeta_0])
                    Ki = K(Xr,Xi,gamma_r,gamma_s,M,br,kr,r1)
                    Kc = K(Xr,Xc,gamma_r,gamma_s,M,br,kr,r1)
                    Ko = K(Xr,Xo,gamma_r,gamma_s,M,br,kr,r1)
                    A = (Ki-2*Kc+Ko)/(2*e**2)
                    B = (Ko-Ki)/(2*e)
                    C = Kc
                    # Determine if planar or non-planar I_ij definition should be used
                    if abs(zeta_0)<1e-6:
                        I_ij = (eta_0**2*A+eta_0*B+C)*(1./(eta_0-e)-1./(eta_0+e))+\
                            (B/2+eta_0*A)*np.log(((eta_0-e)/(eta_0+e))**2)
                    else:
                        I_ij = ((eta_0**2-zeta_0**2)*A+eta_0*B+C)*zeta_0**(-1)*\
                            np.arctan(2*e*abs(zeta_0)/(r1**2-e**2))+\
                            (B/2+eta_0*A)*np.log((r1**2-2*eta_0*e+e**2)/\
                            (r1**2+2*eta_0*e+e**2))+2*e*A
                    D[i,j]+=delta_x_j*np.cos(lambda_j)/(8.*np.pi)*I_ij
            #Area[i,i] = recievingBox.Area
            # Assemble total derivative matrix [W]
            for NID, factor in recievingBox.DOF.iteritems():
                col = self.nodeDict[NID]
                W[i,col*6+2] = -1j*kr/br*factor
                W[i,col*6+4] = (1.+1j*kr/br*recievingBox.xarm)*factor
            '''if not -1 in recievingBox.DOF.keys():
                nids = recievingBox.DOF.keys()
                col1 = self.nodeDict[nids[0]]
                W[i,col1*6+2] = -1j*kr/br*recievingBox.DOF[nids[0]]
                W[i,col1*6+4] = 1j*kr/br*recievingBox.xarm*recievingBox.DOF[nids[0]]
                col2 = self.nodeDict[nids[1]]
                W[i,col2*6+2] = -1j*kr/br*recievingBox.DOF[nids[1]]
                W[i,col2*6+4] = 1j*kr/br*recievingBox.xarm*recievingBox.DOF[nids[1]]'''
        # Create integration matrix [B]
        B = br/kr*np.dot(np.imag(W.T),self.AeroArea)
        self.D = D
        self.W = W
        chd = np.dot(B,np.dot(np.linalg.inv(D),W))
        self.Qaic = chd
        
        # Determine the list of NIDs to be contrained
        cnds = sorted(list(self.const.keys()))
        # Initialize the number of equations to be removed from the system
        deleqs = 0
        # For the number of constrained NIDs
        for i in range(0,len(self.const)):
            # The row range to be removed associated with the NID
            row = self.nodeDict[cnds[i]]
            # Determine which DOF are to be removed
            tmpcst = self.const[cnds[i]]
            # For all of the degrees of freedom to be removed
            for j in range(0,len(tmpcst)):
                # Remove the row associated with the jth DOF for the ith NID
                chd = np.delete(chd,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                chd = np.delete(chd,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                # Incremend the number of deleted equations
                deleqs += 1
        # Save the reduced global stiffness matrix
        self.Qaicr = chd
            
                

class LoadSet:
    def __init__(self,LID):
        self.LID = LID
        self.pointLoads = {}
        self.distributedLoads = {}
    def addPointLoad(self,F,NID):
        if NID in self.pointLoads.keys():
            self.pointLoads[NID]=self.pointLoads[NID]+F
        else:
            self.pointLoads[NID]=F
    def addDistributedLoad(self,f,eid):
        if eid in self.distributedLoads.keys():
            self.distributedLoads[eid]=self.distributedLoads[eid]+f
        else:
            self.distributedLoads[eid]=f
            
class FlutterPoint:
    def __init__(self,FPID,U_vec,nModes):
        self.FPID = FPID
        self.U_vec = U_vec
        self.omegaAeroDict = {}
        self.omegaRootDict = {}
        self.gammaDict = {}
        self.shapeDict = {}
        self.omega = []
        self.gamma = []
        self.shape = np.zeros((nModes,len(U_vec)))
        for U in U_vec:
            self.omegaAeroDict[U] = []
            self.omegaRootDict[U] = []
            self.gammaDict[U] = []
            self.shapeDict[U] = []
    def saveSol(self,U,omega_aero,omega_root,gamma_root,shape):
        if U not in self.U_vec:
            raise ValueError('A velocity is being written that wasnt provided')
        self.omegaAeroDict[U] += [omega_aero]
        self.omegaRootDict[U] += [omega_root]
        self.gammaDict[U] += [gamma_root]
        self.shapeDict[U] += [shape]
    def interpOmegaRoot(self):
        i = 0
        for U in self.U_vec:
            omegaAeros = self.omegaAeroDict[U]
            omegaRoots = self.omegaRootDict[U]
            gammaRoots = self.gammaDict[U]
            shapes = self.shapeDict[U]
            ind1 = 'none'
            omegaDiff = np.array(omegaAeros)-np.array(omegaRoots)
            for k in range(0,len(omegaAeros)-1):
                if omegaDiff[k]*omegaDiff[k+1]<0:
                    ind1 = k
                    break
            if ind1=='none':
                print(U)
                print(omegaAeros)
                print(omegaRoots)
                raise ValueError("Omega_aero never equals omega_root"
                    "for mode %d at airspeed %4.2f . Consider adding more reduces "
                    "frequency values." %(self.FPID,U))
            # Linearly interpolating between frquencies, determine
            # parameter t which can be used to interpolate frequencies
            omega_aero1 = omegaAeros[ind1]
            omega_aero2 = omegaAeros[ind1+1]
            omega_root1 = omegaRoots[ind1]
            omega_root2 = omegaRoots[ind1+1]
            gamma_root1 = gammaRoots[ind1]
            gamma_root2 = gammaRoots[ind1+1]
            shape_root1 = np.array(shapes[ind1])
            shape_root2 = np.array(shapes[ind1+1])
            true_mode_omega = -(omega_aero2*omega_root1-omega_aero1*omega_root2)\
                /(omega_aero1-omega_aero2-omega_root1+omega_root2)
            true_mode_gamma = ((gamma_root2-gamma_root1)/(omega_aero2-omega_aero1))\
                *(true_mode_omega-omega_aero1)+gamma_root1
            true_mode_shape = (shape_root2-shape_root1)/(omega_aero2-omega_aero1)\
                *(true_mode_omega-omega_aero1)+shape_root1
            '''t = (omega_aero1-omega_root1)/((omega_root2-omega_root1)-\
                (omega_aero2-omega_aero1))
            true_mode_omega = omega_aero1+t*(omega_aero2-omega_aero1)
            true_mode_gamma = gamma_root1+t*(gamma_root2-gamma_root1)
            true_mode_shape = shape_root1+t*(shape_root2-shape_root1)'''
            '''# Temporary dimension check for mode_shape
            if not np.size(true_mode_shape,axis=1)==1:
                array_height = np.size(true_mode_shape,axis=0)
                array_length = np.size(true_mode_shape,axis=1)
                return ValueError('true_mode_shape numpy arrays shape is:'+
                    '%dx%d' %(array_height,array_length))'''
            self.omega += [true_mode_omega/(np.pi*2)]
            self.gamma += [true_mode_gamma]
            self.shape[:,i] = true_mode_shape
            i+=1