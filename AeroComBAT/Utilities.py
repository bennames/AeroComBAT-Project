# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np
from tabulate import tabulate

class RotationHelper:
    def transformCompl(self,S,th,**kwargs):
        xsect = kwargs.pop('xsect',False)
        Sxsect = S
        if xsect:
            # Note: This operation is done because during material property input,
            # the convention is that the fibers of a composite run in the
            # x direction, however for cross-sectional analysis, the nominal fiber
            # angle is parallel to the z-axis, and so before any in plane rotations
            # occur, the fiber must first be rotated about the y axis.
            Rysig, Ryeps = self.genCompRy(90)
            Sxsect = np.dot(Ryeps,np.dot(Sxsect,np.linalg.inv(Rysig)))
        # Rotate material about x:
        Rxsig, Rxeps = self.genCompRx(th[0])
        Sxsect = np.dot(Rxeps,np.dot(Sxsect,np.linalg.inv(Rxsig)))
        # Rotate material about x:
        Rysig, Ryeps = self.genCompRy(th[1])
        Sxsect = np.dot(Ryeps,np.dot(Sxsect,np.linalg.inv(Rysig)))
        # Rotate material about x:
        Rzsig, Rzeps = self.genCompRz(th[2])
        Sxsect = np.dot(Rzeps,np.dot(Sxsect,np.linalg.inv(Rzsig)))
        return Sxsect
    def rotXYZ(self,th):
        th = np.deg2rad(th)
        Rx = np.array([[1.,0.,0.],\
                       [0.,np.cos(th[0]),-np.sin(th[0])],\
                       [0.,np.sin(th[0]),np.cos(th[0])]])
        Ry = np.array([[np.cos(th[1]),0.,np.sin(th[1])],\
                       [0.,1.,0.],\
                       [-np.sin(th[1]),0.,np.cos(th[1])]])
        Rz = np.array([[np.cos(th[2]),-np.sin(th[2]),0.],\
                       [np.sin(th[2]),np.cos(th[2]),0.],\
                       [0.,0.,1.]])
        return np.dot(Rz,np.dot(Ry,Rx))
    def genRotMat(self,a,b):
        if all(a==b):
            return np.eye(3)
        else:
            v = np.cross(a,b)
            s = np.linalg.norm(v)
            c = np.dot(a,b)
            vstar = np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
            return np.eye(3)+vstar+np.dot(vstar,vstar)*(1-c)/s**2
    def genCompRx(self,th):
        th = np.deg2rad(th)
        s = np.sin(th)
        c = np.cos(th)
        Rxsig = np.array([[1.,0.,0.,0.,0.,0.],\
                          [0.,c**2,s**2,2*c*s,0.,0.],\
                          [0.,s**2,c**2,-2*c*s,0.,0.],\
                          [0.,-c*s,c*s,c**2-s**2,0.,0.],\
                          [0.,0.,0.,0.,c,-s],\
                          [0.,0.,0.,0.,s,c]])
        Rxeps = np.array([[1.,0.,0.,0.,0.,0.],\
                          [0.,c**2,s**2,c*s,0.,0.],\
                          [0.,s**2,c**2,-c*s,0.,0.],\
                          [0.,-2*c*s,2*c*s,c**2-s**2,0.,0.],\
                          [0.,0.,0.,0.,c,-s],\
                          [0.,0.,0.,0.,s,c]])
        return Rxsig, Rxeps
    def genCompRy(self,th):
        th = np.deg2rad(th)
        s = np.sin(th)
        c = np.cos(th)
        Rysig = np.array([[c**2,0.,s**2,0.,2*c*s,0.],\
                          [0.,1.,0.,0.,0.,0.],\
                          [s**2,0.,c**2,0.,-2*c*s,0.],\
                          [0.,0.,0.,c,0.,-s],\
                          [-c*s,0.,c*s,0.,c**2-s**2,0.],\
                          [0.,0.,0.,s,0.,c]])
        Ryeps = np.array([[c**2,0.,s**2,0.,c*s,0.],\
                          [0.,1.,0.,0.,0.,0.],\
                          [s**2,0.,c**2,0.,-c*s,0.],\
                          [0.,0.,0.,c,0.,-s],\
                          [-2*c*s,0.,2*c*s,0.,c**2-s**2,0.],\
                          [0.,0.,0.,s,0.,c]])
        return Rysig, Ryeps
    def genCompRz(self,th):
        th = np.deg2rad(th)
        s = np.sin(th)
        c = np.cos(th)
        Rzsig = np.array([[c**2,s**2,0.,0.,0.,2*c*s],\
                          [s**2,c**2,0.,0.,0.,-2*c*s],\
                          [0.,0.,1.,0.,0.,0.],\
                          [0.,0.,0.,c,s,0.],\
                          [0.,0.,0.,-s,c,0.],\
                          [-c*s,c*s,0.,0.,0.,c**2-s**2]])
        Rzeps = np.array([[c**2,s**2,0.,0.,0.,c*s],\
                          [s**2,c**2,0.,0.,0.,-c*s],\
                          [0.,0.,1.,0.,0.,0.],\
                          [0.,0.,0.,c,s,0.],\
                          [0.,0.,0.,-s,c,0.],\
                          [-2*c*s,2*c*s,0.,0.,0.,c**2-s**2]])
        return Rzsig, Rzeps
    def getEulerAxisRotMat(self,a,th):
        astar = np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
        a = np.reshape(a,(3,1))
        R = np.cos(th)*np.eye(3)+(1-np.cos(th))*np.dot(a,a.T)+np.sin(th)*astar
        return R
    def getTransMat(self,x):
        return np.array([[1.,0.,0.,-x[0]],\
                         [0.,1.,0.,-x[1]],\
                         [0.,0.,1.,-x[2]],\
                         [0.,0.,0.,1.]])
