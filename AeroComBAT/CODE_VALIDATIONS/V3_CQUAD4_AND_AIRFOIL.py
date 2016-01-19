# =============================================================================
# HEPHAESTUS VALIDATION 3 - CQUAD4 AND AIRFOIL
# =============================================================================

# Import Statements
from Structures import Node, MaterialLib, CQUAD4
from AircraftParts import Airfoil
import numpy as np
import pylab as pl

# Material Info
mat_lib = MaterialLib()
mat_lib.addMat(1, 'AL-2050', 'iso', \
                [75.8, 0.33, 2.7e3], .15e-3)
                
# =============================================================================
# CQUAD4 2D ELEMENT VALIDATION
# =============================================================================
n1 = Node(1,[0.,0.,0.])
n2 = Node(2,[2.,0.,0.])
n3 = Node(3,[2.,3.,0.])
n4 = Node(1,[0.,5.,0.])
elem1 = CQUAD4(1,[n1,n2,n3,n4],1,mat_lib)
elem1.printSummary()

# =============================================================================
# AIRFOIL OUTER MOLD LINE VALIDATION
# =============================================================================
c = .01
af1 = Airfoil(c,name='box')
x = np.linspace(-.5,.5,500)
xu,yu,xl,yl = af1.points(x)
pl.figure(num=1)
pl.plot(xu,yu)
pl.hold(True)
pl.plot(xl,yl)
pl.hold(False)
pl.axes().set_aspect('equal', 'datalim')



c = .01
af2 = Airfoil(c,name='NACA2412')
x = np.linspace(0,1.,500)
xu,yu,xl,yl = af2.points(x)
pl.figure(num=2)
pl.plot(xu,yu)
pl.hold(True)
pl.plot(xl,yl)
pl.hold(False)
pl.axes().set_aspect('equal', 'datalim')