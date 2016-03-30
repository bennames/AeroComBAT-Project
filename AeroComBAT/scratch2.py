# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03 18:52:53 2016

@author: Ben
"""
#from Aerodynamics import CAERO1
from Structures import MaterialLib
from AircraftParts import Wing
import numpy as np
from FEM import Model

# Define the width of the cross-section
x1 = .2
x2 = .6
c = 2
ctip = c
croot = c
p1 = np.array([0.,0.,0.])
p2 = np.array([0.,20,0.])
Y_rib = np.linspace(0.,1.,2)
b_s = np.linalg.norm((Y_rib[0],Y_rib[-1]))

matLib = MaterialLib()
matLib.addMat(1,'AL','iso',[71.7e9,.33,2810],.005)
matLib.addMat(2,'Weak_mat','iso',[100,.33,10],.005)

n_ply = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
m_i = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

noe_dens = 4
wing1 = Wing(1,p1,p2,croot,ctip,x1,x2,Y_rib,n_ply,m_i,matLib,name='NACA0012',noe=noe_dens,ref_ax='origin')

x1 = np.array([0.,0.,0.])
x2 = np.array([c,0.,0.])
x3 = np.array([c,20.,0.])
x4 = np.array([0.,20.,0.])
nspan = 10
nchord = 2

wing1.addLiftingSurface(1,x1,x2,x3,x4,nspan,nchord)

#wing1.plotRigidWing(numXSects=10)

model = Model()

model.addAircraftParts([wing1])

model.plotRigidModel(numXSects=10)

from Aerodynamics import K
import multiprocessing as mp

box1 = model.aeroBox[0]

Xr = box1.Xr
Xi = box1.Xi
Xc = box1.Xc
Xo = box1.Xo

gamma_r = box1.dihedral
gamma_s = box1.dihedral

M = .24

kr = .46

br = c/2

r1=0.

output = mp.Queue()

def multi_K(Xr,Xs,gamma_r,gamma_s,M,br,kr,r1,output):
    output.put(K(Xr,Xs,gamma_r,gamma_s,M,br,kr,r1))

Pi = mp.Process(target=multi_K,args=(Xr,Xi,gamma_r,gamma_s,M,br,kr,r1,output))
Pc = mp.Process(target=multi_K,args=(Xr,Xc,gamma_r,gamma_s,M,br,kr,r1,output))
Po = mp.Process(target=multi_K,args=(Xr,Xo,gamma_r,gamma_s,M,br,kr,r1,output))

processes = [Pi,Pc,Po]

for p in processes:
    p.start()
for p in processes:
    p.join()
print(output.empty())
Ki = output.get()
Kc = output.get()
Ko = output.get()

#panel1 = CAERO1(1,x1,x2,x3,x4,nspan,nchord)
#panel1.plotLiftingSurface()
