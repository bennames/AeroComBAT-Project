# =============================================================================
# HEPHAESTUS VALIDATION 4 - MESHER AND CROSS-SECTIONAL ANALYSIS
# =============================================================================

# IMPORTS:

import sys
import os

sys.path.append(os.path.abspath('..\..'))

import mayavi.mlab as mlab
import numpy as np
import cProfile

x = np.linspace(0,1,25)
y = np.linspace(1,3,25)
x,y = np.meshgrid(x,y)
z = np.random.rand(25,25)*1e-1
c = x**2+np.cos(y*z)
z = c

vmin = np.min(z)
vmax = np.max(z)

def plotElements():
    for i in range(25-1):
        for j in range(25-1):
            xtmp = x[j:j+2,i:i+2]
            ytmp = y[j:j+2,i:i+2]
            ztmp = z[j:j+2,i:i+2]
            ctmp = c[j:j+2,i:i+2]
            mlab.mesh(xtmp,ytmp,ztmp,scalars=ctmp,vmin=vmin,vmax=vmax)
            mlab.mesh(xtmp,ytmp,ztmp,representation='wireframe',color = (0,0,0))
            
def plotSurface():
    mlab.mesh(x,y,z,scalars=c)
    
# Time Plot Elements
mlab.figure(figure=1)
cProfile.run('plotElements()',sort='cumtime')
# Plot Suraface
#mlab.figure(figure=2)
#cProfile.run('plotSurface()')

import vtk

