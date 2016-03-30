# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================
from Structures import WingSection, Laminate
from Aerodynamics import CAERO1
# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np
import mayavi.mlab as mlab
# =============================================================================
# WING OPTIMIZATION CLASS
# =============================================================================
class Wing:
    def __init__(self,PID,p1,p2,croot,ctip,x0_spar,xf_spar,Y_rib,n_ply,m_ply,mat_lib,**kwargs):
        """Creates a wing object.
        
        This object represents a wing and contains both structural and
        aerodynamic models.
        
        :Args:
        - `p1 (1x3 np.array[float])`: The initial x,y,z coordinates of the wing.
        - `p2 (1x3 np.array[float]`: The final x,y,z coordinates of the wing.
        - `croot (float)`: The root chord length.
        - `ctip (float)`: The tip chord length.
        - `x0_spar (float)`: The non-dimensional starting location of the cross
            section.
        - `xf_spar (float)`: The non-dimensional ending location of the cross
            section.
        - `Y_rib (1xN Array[float])`: The non-dimensional rib locations within
            the wing. This dimension is primarily used to create wing-sections
            which primarily define the buckling span's for laminate objects.
        - `n_ply (1xM Array[int])`: An array of integer's specifying the number
            plies to be used in the model. Each integer refers to the number of
            plies to be used for at a given orientation.
        - `m_ply (1xM Array[int])`: An array of integer's specifying the
            material ID to be used for the corresponding number of plies in
            n_ply at a given orientation.
        - `mat_lib (obj)`: A material library containing all of the material
            objets to be used in the model.
        - `PID (int)`: The integer ID for the wing part object.
        - `name (str)`: The name of the airfoil section to be used for cross
            section generation.
        - `wing_SNID (int)`: The first node ID associated with the wing.
        - `wing_SEID (int)`: The first beam element ID associated with the wing.
        - `wing_SSBID (int)`: The first superbeam ID associated with the wing.
        - `SXID (int)`: The starting cross-section ID used by the wing.
        - `noe (float)`: The number of beam elements to be used in the wing per
            unit length.
        - `n_orients (int)`: The number of fiber orientations to be used in
            each laminate.
        - `n_lams (int)`: The number of laminates required to mesh the desired
            cross-section.
        - `meshSize (float)`: The maximum aspect ratio a 2D element may have in
            the cross-section.
        - `ref_ax (str)`: The reference axis to be loaded in the wing.
        """
        #The type of the object
        self.type='wing'
        # Initialize the array holding wing sections
        self.wingSects = []
        # Initialize the wing ID
        self.PID = PID
        # Initialize Lifting surface Array
        self.liftingSurfaces = {}
        # Name of the airfoil section (used to generate the OML shape of the x-section)
        name = kwargs.pop('name','NACA0012')
        # The initial starting node ID for the structural generation of the wing
        tmp_SB_SNID = kwargs.pop('wing_SNID',0)
        # The initial beam element EID for the first superbeam ID
        tmp_SB_SEID = kwargs.pop('wing_SEID',0)
        # The initial starting superbeam ID
        tmp_SB_SBID = kwargs.pop('wing_SSBID',0)
        # The starting cross-section ID
        SXID = kwargs.pop('SXID',0)
        # The number of beam elements to be used per unit length
        noe = kwargs.pop('noe',10)
        # The number of fiber orientations to be used in each laminate
        n_orients = kwargs.pop('n_orients',4)
        # The number of laminates required to mesh the desired cross-section
        n_lams = kwargs.pop('n_lams',4)
        # Laminate symmetry
        lam_sym = kwargs.pop('lam_sym',False)
        # The maximum aspect ratio a 2D element may have in the cross-section
        meshSize = kwargs.pop('meshSize',4)
        # The reference axis to be loaded in the wing
        ref_ax = kwargs.pop('ref_ax','shearCntr')
        # Chord vector for wing
        chordVec = kwargs.pop('chordVec',np.array([1.,0.,0.]))
        # Orientations of each ply in the lamiante
        th_ply = kwargs.pop('th_ply',[0]*len(n_ply))
        # Type of cross-section
        typeXSect = kwargs.pop('typeXSect','box')
        # Calculate the wing span:
        b_s = np.linalg.norm(p2-p1)
        # Lambda function to calculate average panel chord length on on the fly.
        chord = lambda y: (ctip-croot)*y/b_s+croot
        # Create wing sections between each rib:
        for i in range(0,len(Y_rib)-1):
            # Create a wing panel object based on the average chord length
            # Determine the laminate schedule beam section
            section_lams = []
            for j in range(0,n_lams):
                # Select vectors of thicknesses and MID's:
                n_i_tmp = n_ply[i*n_lams+n_orients*j:i*n_lams+n_orients*j+n_orients]
                m_i_tmp = m_ply[i*n_lams+n_orients*j:i*n_lams+n_orients*j+n_orients]
                th_i_tmp = th_ply[i*n_lams+n_orients*j:i*n_lams+n_orients*j+n_orients]
                section_lams += [Laminate(n_i_tmp,m_i_tmp,mat_lib,sym=lam_sym,th=th_i_tmp)]
            # Compile all information needed to create xsection and beams
            # Starting coordiante of super beam
            tmp_x1 = p1+Y_rib[i]*(p2-p1)
            # Ending coordiante of super beam
            tmp_x2 = p1+Y_rib[i+1]*(p2-p1)
            tmpWingSect = WingSection(tmp_x1,tmp_x2,chord,name,x0_spar,xf_spar,\
                section_lams,mat_lib,noe,SSBID=tmp_SB_SBID,SNID=tmp_SB_SNID,\
                SEID=tmp_SB_SEID,meshSize=meshSize,SXID=SXID,ref_ax=ref_ax,\
                chordVec=chordVec,typeXSect=typeXSect)
            # Prepare ID values for next iteration
            tmp_SB_SNID = tmpWingSect.SuperBeams[-1].enid
            tmp_SB_SEID = max(tmpWingSect.SuperBeams[-1].elems.keys())+1
            tmp_SB_SBID = tmpWingSect.SuperBeams[-1].SBID+1
            self.wingSects += [tmpWingSect]
            SXID = max(self.wingSects[i].XIDs)+1
            #self.model.addElements(tmpWingSect.SuperBeams)
    def addLiftingSurface(self,SID,x1,x2,x3,x4,nspan,nchord):
        # Create the lifting surface
        tmpLiftSurf = CAERO1(SID,x1,x2,x3,x4,nspan,nchord)
        # Create a temporary dictionary of CQUADA's to iterate through later
        Dict = tmpLiftSurf.CQUADAs.copy()
        # Store it in the wing object
        self.liftingSurfaces[SID]=tmpLiftSurf
        # CONNECT AERO BOXES TO ELEMENTS
        # For all elements in the wing
        for wingSect in self.wingSects:
            for superBeam in wingSect.SuperBeams:
                for EID, elem in superBeam.elems.iteritems():
                    # For all panels in the lifting surface
                    tmpDict = Dict.copy()
                    for PANID, panel in tmpDict.iteritems():
                        # Determine the y-coord of the recieving point
                        y34pan = panel.y(-0.5,0)
                        # If the panel y-coord is between the nodes of the elem
                        if (y34pan-elem.n1.x[1])*(y34pan-elem.n2.x[1])<0:
                            # The panel's displacement can be described by the
                            # displacements of the nodes used by the element.
                            # Determine the x-coord in the elem corresponding
                            # to the location of the panel recieving point
                            t = (y34pan-elem.n1.x[1])/(elem.n2.x[1]-elem.n1.x[1])
                            # Determine the nodal contributions of the displacements
                            panel.DOF[elem.n1.NID] = t
                            panel.DOF[elem.n2.NID] = 1-t
                            # Determine the moment arm of the box acting on the
                            # beam
                            x34elem = elem.n1.x[0]+t*(elem.n2.x[0]-elem.n1.x[0])
                            x34pan = panel.x(-0.5,0)
                            panel.xarm = x34pan-x34elem
                            # Remove this reference from the dictionary so it
                            # is not iterated over again for the next element
                            del Dict[PANID]
                for NID, node in superBeam.nodes.iteritems():
                    # For all panels in the lifting surface
                    tmpDict = Dict.copy()
                    for PANID, panel in tmpDict.iteritems():
                        # Determine the y-coord of the recieving point
                        y34pan = panel.y(-0.5,0)
                        # If the panel y-coord is between the nodes of the elem
                        if abs(y34pan-node.x[1])<1e-6:
                            # The panel's displacement can be described by the
                            # displacements of the nodes used by the element.
                            # Determine the x-coord in the elem corresponding
                            # to the location of the panel recieving point
                            t = 1
                            # Determine the nodal contributions of the displacements
                            panel.DOF[node.NID] = t
                            # Determine the moment arm of the box acting on the
                            # beam
                            x34node = node.x[0]
                            x34pan = panel.x(-0.5,0)
                            panel.xarm = x34pan-x34node
                            # Remove this reference from the dictionary so it
                            # is not iterated over again for the next element
                            del Dict[PANID]
        if len(Dict.keys())>0:
            print('Warning, some elements could not have their displacements'+
                ' matched by beam elements. This includes:')
        for PANID, panel in Dict.iteritems():
            #panel.DOF[-1] = None
            print('CQUADA %d' %(PANID))
    def plotRigidWing(self,**kwargs):
        figName = kwargs.pop('figName','Rigid Wing')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        mlab.figure(figure=figName)
        for sects in self.wingSects:
            sects.plotRigid(figName=figName,clr=clr,numXSects=numXSects)
        if len(self.liftingSurfaces)>0:
            for SID, surface in self.liftingSurfaces.iteritems():
                surface.plotLiftingSurface(figName=figName)
    '''def addConstraint(self,NID,const):
        #INPUTS:
        #nid - The node you want to constrain
        #const - a numpy array containing integers from 1-6 or a string description.
        #For example, to pin a beam in all three directions, the const array would look like:
        # const = np.array([1,2,3],dtype=int) = 'pin'
        #A fully fixed node constraint would look like:
        # const = np.array([1,2,3,4,5,6],dtype=int) = 'fix'
        self.model.applyConstraints(NID,const)
    def applyLoads(self,**kwargs):
        #INPUTS
        #f - a function taking the form of:
        #def f(x):
        #   vx = (1/10)*10*x[2]**2-7*x[2]-2.1
        #   vy = 10*x[2]**2-7*x[2]
        #   pz = 0
        #   tz = (10*x[2]**2-7*x[2])/10+3*x[0]**2
        #   return np.array([vx,vy,pz,tz])
        #This is in other words a function describing the distributed loads in
        #beam as a function of GLOBAL position.
        #
        #eid - The element id that the distributed loads are to be applied to.
        #
        #F - a dictionary taking the form of:
        #F[node id] = np.array([Qx,Qy,P,Mx,My,T])
        #TODO: Make it so that local CSYS can be used for load applications. This
        #Should allow for translation and rotations.
        # Get the distributed load function
        f = kwargs.pop('f',None)
        F = kwargs.pop('F',None)
        eid = kwargs.pop('eid',[])
        allElems = kwargs.pop('allElems',False)
        if f==None:
            self.model.applyLoads(eid=eid,F=F,allElems=allElems)
        else:
            self.model.applyLoads(eid=eid,f=f,F=F,allElems=allElems)
        
    def resetPointLoads(self):
        self.model.resetPointLoads()
    def staticAnalysis(self,resetPointLoads=False,analysis_name='analysis_untitled'):
        # Reset any results data contained within the cross-section objects
        # such as curvatures at nodes and stresses/strains within planer
        # xsection elements
        for wingSect in self.wingSects:
            for sbeam in wingSect.SuperBeams:
                sbeam.xsect.resetResults()
        # Reset the point loads in the model
        if resetPointLoads:
            self.model.resetPointLoads()
        # Run the FEM static analysis
        self.model.staticAnalysis(analysis_name=analysis_name)
    def normalModesAnalysis(self,analysis_name='analysis_untitled'):
        # Run normal modes analysis
        self.model.normalModesAnalysis(analysis_name=analysis_name)
    def plotRigidWing(self,**kwargs):
        figName = kwargs.pop('figName','Rigid Wing')
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
        # Chose the color of the beam, defaults to black, accepts tuple
        clr = kwargs.pop('color',(0,0,0))
        # Chose the number of cross-sections to be plotted. By default this is 2
        # One at the beggining and one at the end of the super beam
        numXSects = kwargs.pop('numXSects',2)
        if environment=='mayavi':
            mlab.figure(figure=figName)
            for sects in self.wingSects:
                sects.plotRigid(figName=figName,clr=clr,numXSects=numXSects)
    def plotDeformedWing(self,**kwargs):
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        figName = kwargs.pop('figName','Deformed Wing')
        # Select the plotting environment you'd like to choose
        environment = kwargs.pop('environment','mayavi')
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
        if environment=='mayavi':
            mlab.figure(figure=figName)
            for sects in self.wingSects:
                sects.plotDispl(figName=figName,clr=clr,numXSects=numXSects,\
                    contLim=contLim,warpScale=warpScale,displScale=displScale,\
                    contour=contour,analysis_name=analysis_name,mode=mode)
        mlab.colorbar()'''