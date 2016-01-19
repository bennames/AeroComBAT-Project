from Structures import Node
from Structures import CQUAD4

import numpy as np


class Mesher:
    def boxBeam(xsect,meshSize,x0,xf,nodeDict,elemDict,matlib,**kwargs):
        # INITIALIZE INPUTS
        # The laminates used to mesh the cross-seciton
        laminates = xsect.laminates
        # Initialize the airfoil
        Airfoil = xsect.airfoil
        # the chord length of the airfoil profile
        c = Airfoil.c
        # Initialize the z location of the cross-section
        zc = kwargs.pop('zc',0)
        # Initialize the Euler angler rotation of the fiber materials for
        # any the given laminate. Note that futher local element material
        # properties can occur if there is curving for example in the OML
        # of the cross-section.
        thz = [0,90,180,270]
        
        # CREATE NODES FOR MESH
        # Verify that 4 laminate objects have been provides
        if not len(laminates)==4:
            raise ValueError('The box beam cross-section was selected, but 4 laminates were not provided')
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
        #TODO: Make meshing laminates more efficient so that code can be reused
        
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
                        MID = xsect.laminates[k].plies[i].MID
                        th = [0,xsect.laminates[k].plies[i].th,thz[k]+thz_loc]
                        #print('Ply %d' %(i+1))
                        #print('Element %d' %(j+1))
                    # Else if it is vertical:
                    else:
                        MID = xsect.laminates[k].plies[j].MID
                        th = [0,xsect.laminates[k].plies[j].th,thz[k]]
                        #print('Ply %d' %(j+1))
                        #print('Element %d' %(i+1))
                    #for node in nodes:
                        #node.printNode()
                    #print('\n')
                    elemDict[newEID] = CQUAD4(newEID,nodes,MID,matlib,th=th)
                    xsect.laminates[k].EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]
    def cylindricalTube(xsect,r,meshSize,x0,xf,nodeDict,elemDict,matlib,**kwargs):
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
                    elemDict[newEID] = CQUAD4(newEID,nodes,MID,matlib,th=th)
                    xsect.lam.EIDmesh[i,j] = newEID
        xsect.elemDict = elemDict
        del xsect.nodeDict[-1]
        del xsect.elemDict[-1]