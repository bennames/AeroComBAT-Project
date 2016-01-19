#Author: Ben Names
#Created on: 2015-04-03
#A part of package: SEAS
#Version: V 0.1
#==============================================================================
# IMPORT STATEMENTS
#==============================================================================
import numpy as np
import scipy as sci
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#%% BEAM FEM CLASS
class Model:
    def __init__(self):
        # Global Stiffness Matrix
        self.Kg = None
        # Reduced Global Stiffness Matrix
        self.Kgr = None
        # Global Force Vector
        self.Fg = None
        # Global Reduced Force Matricies
        self.Fgr = None
        # Post-processed Force Vector
        self.Fpp = None
        # Force Boundary Conditions
        self.Qg = None
        # List of node ID's contained in the model
        self.nids = []
        # Dictionary Mapping Node ID's to the global stiffness matrix
        self.nodeDict = {}
        # Numpy Object Array Containing all of the Elements in the Global System
        self.elems = []
        # Dictionary Mapping Node ID's to restricted degrees of Freedom
        self.const = {}
        # Array of Displacements
        self.u = None
        # Analysis ID
        self.AID=0
    def addElements(self, elemarray):
        # For all of the terms in the array elemarray
        for i in range(0,len(elemarray)):
            # If the ith term is a superbeam
            if elemarray[i].type=='SuperBeam':
                # Get all of the elements in the superbeam
                SBeam = elemarray[i]
                for EID, elem in SBeam.elems.iteritems():
                    self.elems += [elem]
                    self.nids += [elem.n1.NID,elem.n2.NID]
            elif elemarray[i].type=='Tbeam':
                TBeam = elemarray[i]
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
    def resetPointLoads(self):
        self.Qg = np.zeros(6*len(self.nids))
    def assembleGlobalModel(self,static4BuckName = 'analysis_untitled'):
        # Determine the degrees of freedom in the model
        DOF = 6*len(self.nids)
        # Initialize the global stiffness matrix
        Kg = np.zeros((DOF,DOF),dtype=float)
        # Initialize the global force vector
        Fg = np.zeros((DOF,1),dtype=float)
        # Initialize the geometric global stiffness matrix
        Kgg = np.zeros((DOF,DOF),dtype=float)
        # Initialize the global mass matrix
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
                # Add the elem force vector to the global matrix
                Fg[6*row:6*row+6,:] = Fg[6*row:6*row+6,:] + elem.Fe[6*i:6*i+6,:] \
                                    + np.reshape(self.Qg[6*row:6*row+6],(6,1))
                for j in range(0,len(nodes)):
                    # Determine the column range for the NID
                    col = self.nodeDict[nodes[j]]
                    # Add the elem stiffness matrix portion to the global
                    # stiffness matrix
                    Kg[6*row:6*row+6,6*col:6*col+6] = Kg[6*row:6*row+6,6*col:6*col+6]\
                                                    +elem.Ke[6*i:6*i+6,6*j:6*j+6]
                    # Determine the axial force in the beam
                    if static4BuckName in elem.F1.keys():
                        Ploc = elem.F1[static4BuckName][2]
                    else:
                        Ploc=0.
                    # Add the elem geometric stiffness matrix portion to the
                    # global stiffness matrix
                    Kgg[6*row:6*row+6,6*col:6*col+6] = Kgg[6*row:6*row+6,6*col:6*col+6]\
                                                    +elem.Keg[6*i:6*i+6,6*j:6*j+6]*Ploc
                    # Add the element mass matrix portion to the global mass matrix
                    Mg[6*row:6*row+6,6*col:6*col+6] = Mg[6*row:6*row+6,6*col:6*col+6]\
                                                    +elem.Me[6*i:6*i+6,6*j:6*j+6]
        # Save the global stiffness matrix
        self.Kg = Kg
        # Save the global force vector
        self.Fg = Fg
        # Save the global geometric stiffness matrix
        self.Kgg = Kgg
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
                Fg = np.delete(Fg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                Kg = np.delete(Kg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                Kgg = np.delete(Kgg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                Kgg = np.delete(Kgg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=0)
                Mg = np.delete(Mg,row*6+(tmpcst[j]-1)-deleqs,axis=1)
                # Incremend the number of deleted equations
                deleqs += 1
        # Save the reduced global force vector
        self.Fgr = Fg
        # Save the reduced global stiffness matrix
        self.Kgr = Kg
        # Save the reduced global geometric stiffness matrix
        self.Kggr = Kgg
        # Save the reduced global mass matrix
        self.Mgr = Mg
            
    def applyLoads(self,**kwargs):
        # INPUTS
        # f - a function taking the form of:
        # def f(x):
        #    vx = (1/10)*10*x[2]**2-7*x[2]-2.1
        #    vy = 10*x[2]**2-7*x[2]
        #    pz = 0
        #    tz = (10*x[2]**2-7*x[2])/10+3*x[0]**2
        #    return np.array([vx,vy,pz,tz])
        # This is in other words a function describing the distributed loads in
        #  beam as a function of GLOBAL position.
        #
        # eid - The element id that the distributed loads are to be applied to.
        #
        # F - a dictionary taking the form of:
        # F[node id] = np.array([Qx,Qy,P,Mx,My,T])
        # TODO: Make it so that local CSYS can be used for load applications. This
        # Should allow for translation and rotations.
        def fdefault(x):
                vx = 0.
                vy = 0.
                pz = 0.
                tz = 0.
                return np.array([vx,vy,pz,tz])
        # Get the distributed load function
        f = kwargs.pop('f',fdefault)
        # Get the descrete load dictionary
        F = kwargs.get('F')
        # Keyword to fascilitate applying distributed load to all elements
        allElems = kwargs.pop('allElems',False)
        if allElems:
            eid = [elem.EID for elem in self.elems]
        else:
            # Get the EID's the the load function is to be evaluated at
            eid = kwargs.pop('eid',[])
        if len(eid)==0 and not f==fdefault:
            print('WARNING: You are attempting to apply a distributed load'\
                'without listing EIDs at which to apply those loads.')
        # For all of the EIDs provided
        for ID in eid:
            # For all of the in the model
            for elem in self.elems:
                # If the applied ID is the elem EID
                if elem.EID==ID:
                    if elem.type=='Tbeam':
                        h = elem.h
                        x1 = elem.n1.x
                        x2 = elem.n2.x
                        fx = f((x1+x2)/2)
                        elem.applyDistributedLoad(fx)
                    elif elem.type=='EBbeam':
                        h = elem.h
                        x1 = elem.n1.x
                        x2 = elem.n2.x
                        fx = f((x1+x2)/2)
                        elem.Fe = np.reshape(np.array([h*fx[0]/2.,h*fx[1]/2.,\
                        h*fx[2]/2,-h**2*fx[0]/12.,-h**2*fx[1]/12.,h*fx[3]/2,\
                        h*fx[0]/2.,h*fx[1]/2.,h*fx[2]/2,h**2*fx[0]/12.,\
                        h**2*fx[1]/12.,h*fx[3]/2]),(12,1))
        if F==None:
            pass
        else:
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
                # Rearrange the force vector to be consistent with the DOF (u,v,w,thy.thx.thz)
                Ftmp = np.array([Ftmp[0],Ftmp[1],Ftmp[2],Ftmp[3],Ftmp[4],Ftmp[5]])
                self.Qg[6*self.nodeDict[key]:6*self.nodeDict[key]+6] = \
                self.Qg[6*self.nodeDict[key]:6*self.nodeDict[key]+6]+Ftmp
    def applyConstraints(self,nid,const):
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
            self.const[nid]=np.array([1,2,3],dtype=int)
        elif const=='fix':
            self.const[nid]=np.array([1,2,3,4,5,6],dtype=int)
        else:
            self.const[nid]=const
    def staticAnalysis(self,**kwargs):
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.assembleGlobalModel()
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
        #Not sure what the below is for. Solve reaction forces globaly?
        '''self.Fpp = np.zeros((len(u),1))
        for i in range(0,len(u)):
            self.Fpp[i] = self.Fg[i]+np.dot(self.Kg[i,:],u)'''
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
    def bucklingAnalysis(self,**kwargs):
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.staticAnalysis()
        self.assembleGlobalModel()
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
        # Create Analysis Name
        analysis_name = kwargs.pop('analysis_name','analysis_untitled')
        self.assembleGlobalModel()
        eigs,umode = sci.linalg.eig(self.Kgr,self.Mgr)
        idx = eigs.argsort()
        self.eigs = np.sqrt(np.array(eigs[idx],dtype=float))/(2*np.pi)
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
                umode = np.insert(umode,self.nodeDict[ckeys[i]]*6+(tmpconst[j]-1),np.zeros((1,len(eigs))),axis=0)
        self.umode = umode
        for elem in self.elems:
            if elem.type=='Tbeam':
                nid1 = elem.n1.NID
                nid2 = elem.n2.NID
                Umode1 = umode[6*self.nodeDict[nid1]:6*self.nodeDict[nid1]+6,:]
                Umode2 = umode[6*self.nodeDict[nid2]:6*self.nodeDict[nid2]+6,:]
                elem.saveNodalDispl(Umode1,Umode2,analysis_name=analysis_name)