# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================
from Structures import WingSection, Laminate
from FEM import Model
# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np
import mayavi.mlab as mlab
# =============================================================================
# WING OPTIMIZATION CLASS
# =============================================================================
class Wing:
    def __init__(self,b_s,croot,ctip,x0_spar,x1_spar,Y_rib,n_ply,m_i,mat_lib,**kwargs):
        # Initialize the FEM Model
        self.model = Model()
        # Initialize the array holding wing sections
        self.wingSects = []
        # Initialize the first Super Beam ID
        tmp_SB_SEID = kwargs.pop('tmp_SB_SEID',0)
        # Name of the airfoil section (used to generate the OML shape of the x-section)
        name = kwargs.pop('name','NACA0012')
        #wing_SNID = kwargs.pop('wing_SNID',1)
        tmp_SB_SNID = kwargs.pop('wing_SNID',0)
        tmp_SB_SBID = kwargs.pop('SBID',0)
        noe_density = kwargs.pop('noe_per_unit_length',10)
        n_orients = kwargs.pop('n_orients',4)
        n_lams = kwargs.pop('n_lams',4)
        meshSize = kwargs.pop('meshSize',4)
        sXID = kwargs.pop('sXID',0)
        ref_ax = kwargs.pop('ref_ax','shearCntr')
        # Create a lambda function to calculate average panel chord length on
        # on the fly.
        chord = lambda y: (ctip-croot)*y/b_s+croot
        # Create wing sections between each rib:
        for i in range(0,len(Y_rib)-1):
            # Create a wing panel object based on the average chord length
            # Determine the laminate schedule beam section
            section_lams = []
            for j in range(0,n_lams):
                # Select vectors of thicknesses and MID's:
                n_i_tmp = n_ply[i*n_lams+n_orients*j:i*n_lams+n_orients*j+n_orients]
                m_i_tmp = m_i[i*n_lams+n_orients*j:i*n_lams+n_orients*j+n_orients]
                section_lams += [Laminate(n_i_tmp,m_i_tmp,mat_lib,sym=True)]
            # Compile all information needed to create xsection and beams
            # Starting y-coordiante of super beam
            tmp_x1 = Y_rib[i]
            # Ending y-coordiante of super beam
            tmp_x2 = Y_rib[i+1]
            # Establish tmp SBID as 1+ the max of the last SBID
            #tmp_SBID = max(self.superBeamDict.keys())+1
            #tmp_SB_SEID = max(self.elemDict.keys())+1
            #if len(self.superBeamDict.keys())==1:
            #    tmp_SB_SNID = wing_SNID
            #else:
            #   tmp_SB_SNID = max(self.superBeamDict[tmp_SBID-1].nodes.keys())
            tmpWingSect = WingSection(chord,name,x0_spar,x1_spar,section_lams,\
                mat_lib,tmp_x1,tmp_x2,noe_density,tmp_SB_SBID,snid1=tmp_SB_SNID,\
                sEID=tmp_SB_SEID,meshSize=meshSize,sXID=sXID,ref_ax=ref_ax)
            # Prepare ID values for next iteration
            tmp_SB_SNID = tmpWingSect.SuperBeams[-1].enid
            tmp_SB_SEID = max(tmpWingSect.SuperBeams[-1].elems.keys())+1
            tmp_SB_SBID = tmpWingSect.SuperBeams[-1].SBID
            self.wingSects += [tmpWingSect]
            sXID = max(self.wingSects[i].XIDs)+1
            self.model.addElements(tmpWingSect.SuperBeams)
    def addConstraint(self,NID,const):
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
        mlab.colorbar()