Aerodynamics Module
-------------------

.. automodule:: AeroComBAT.Aerodynamics

DOUBLET-LATTICE KERNEL FUNCTION
+++++++++++++++++++++++++++++++
.. autofunction:: AeroComBAT.Aerodynamics.K

DOUBLET-LATTICE AIC METHOD
++++++++++++++++++++++++++
.. autofunction:: AeroComBAT.Aerodynamics.calcAIC

AIRFOIL
+++++++
.. autoclass:: AeroComBAT.Aerodynamics.Airfoil
   :members: __init__, points, printSummary

CQUADA
++++++
.. autoclass:: AeroComBAT.Aerodynamics.CQUADA
   :members: __init__, x, y, z, J, printSummary

CAERO1
++++++
.. autoclass:: AeroComBAT.Aerodynamics.CAERO1
   :members: __init__, x, y, z, plotLiftingSurface, printSummary