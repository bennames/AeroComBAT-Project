# AircraftParts.py
# Author: Ben Names
"""
This module contains a library of classes devoted to modeling aircraft parts.

The main purpose of this library is to model various types of aircraft parts.
Currently only wing objects are suported, however in the future it is possible
that fuselages as well as other parts will be added.

:SUMARRY OF THE CLASSES:

- `Wing`: Creates a wing aircraft. This wing is capable of modeling the
    structures of the aircraft wing as well as the aerodynamics. The structures
    are modeled with a combination of beam models currently, however it is
    likely that Ritz method laminates will also incorporated for buckling
    prediction purposes. The aerodynamics are currently modeled with potential
    flow doublet panels.

"""
__docformat__ = 'restructuredtext'
# =============================================================================
# AeroComBAT MODULES
# =============================================================================
from Structures import WingSection, Laminate
from Aerodynamics import CAERO1
# =============================================================================
# IMPORT ANACONDA ASSOCIATED MODULES
# =============================================================================
import numpy as np
import mayavi.mlab as mlab
# =============================================================================
# DEFINE AeroComBAT AIRCRAFT PART CLASSES
# =============================================================================

class Wing:
    def __init__(self,PID,p1,p2,croot,ctip,x0_spar,xf_spar,Y_rib,n_ply,m_ply,mat_lib,**kwargs):
        """Creates a wing object.
        
        This object represents a wing and contains both structural and
        aerodynamic models.
        
        :Args:
        
        - `PID (int)`: The integer ID linked to this part.
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
        - `n_ply (1xM Array[int])`: An array of integers specifying the number
            plies to be used in the model. Each integer refers to the number of
            plies to be used for at a given orientation.
        - `m_ply (1xM Array[int])`: An array of integers specifying the
            material ID to be used for the corresponding number of plies in
            n_ply at a given orientation.
        - `th_ply (1xM Array[int])`: An array of floats specifying the
            degree orientations of the plies used by the lamiantes in the
            model.
        - `mat_lib (obj)`: A material library containing all of the material
            objets to be used in the model.
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
        - `chordVec (1x3 np.array[float])`: This numpy array is used to orient
            the cross-section in 3D space. It corresponds to the local unit x-
            vector in the cross-section, expressed in the global frame.
        - `typeXSect (str)`: The type of cross-section to be used by the wing
            structure. Currently the suported typed are 'boxbeam', 'laminate',
            and 'rectBoxBeam'. See the meshing class in the structures module
            for more details.
        
        :Returns:
        
        - None
        
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
        """Adds a potential flow lifting surface to the model.
        
        This method adds a potential flow panel aerodynamic model to the wing
        part. The x1,x2,x3, and x4 points correspond to the root leading edge,
        root trailing edge, tip trailing edge, and tip leading edge of the wing
        respectively. Currently the only suported types of panels are doublet-
        lattice panels to be used for unsteady aerodynamic models.
        
        :Args:
        
        - `SID (int)`: The lifting surface integer identifier corresponding to
            the lifting surface.
        - `x1 (1x3 numpy array)`: The point in 3D space corresponding to the
            root leading edge point of the lifting surface.
        - `x2 (1x3 numpy array)`: The point in 3D space corresponding to the
            root trailing edge point of the lifting surface.
        - `x3 (1x3 numpy array)`: The point in 3D space corresponding to the
            tip trailing edge point of the lifting surface.
        - `x4 (1x3 numpy array)`: The point in 3D space corresponding to the
            tip leading edge point of the lifting surface.
        - `nspan (int)`: The number of boxes to be used in the spanwise
            direction.
        - `nchord (int)`: The number of boxes to be used in the chordwise
            direction.
        
        :Returns:
        
        - None
        
        .. Note:: Mutliple surfaces could be added to the wing part.
        
        .. Warning:: In order to use the doublet lattice method, the chord
            lines of the lifting surface must run in the x-direction, and there
            can be no geometric angles of attack present. The geometry of a
            general wing can be seen in the figure below:
            
        .. image:: images/DoubletLatticeWing.png
            :align: center
        """
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
        """Plots the rigid wing.
        
        This method plots the rigid model of the wing. This includes the
        reference axis of the beams, the cross-sections of the beams, and the
        lifting surfaces that make up the wing. This is an excellent check to
        perform before adding the part to a FEM model.
        
        :Args:
        
        - `figName (str)`: The name of the MayaVi figure. 'Rigid Wing' by
            default.
        - `numXSects (int)`: The number of cross-sections that each wing
            section will display. By default it is 2.
        - `color (1x3 tuple(int))`: This is a length 3 tuple to be used as the
            color of the beam reference axes. Black by default.
        
        :Returns:
        
        - None
        """
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