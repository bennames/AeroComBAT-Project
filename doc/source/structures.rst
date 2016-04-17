Structures Module
-----------------

.. automodule:: AeroComBAT.Structures


NODE
++++

.. autoclass:: AeroComBAT.Structures.Node
   :members: __init__, printSummary

MATERIAL
++++++++

.. autoclass:: AeroComBAT.Structures.Material
   :members: __init__, printSummary

CQUADX
++++++

.. autoclass:: AeroComBAT.Structures.CQUADX
   :members: __init__, x, y, J, resetResults, getDeformed, getStressState, printSummary

MATERIAL LIBRARY
++++++++++++++++

.. autoclass:: AeroComBAT.Structures.MaterialLib
   :members: __init__, addMat, getMat, printSummary

PLY
+++

.. autoclass:: AeroComBAT.Structures.Ply
   :members: __init__, genQ, printSummary

LAMINATE
++++++++

.. autoclass:: AeroComBAT.Structures.Laminate
   :members: __init__, printSummary

MESHER
++++++

.. autoclass:: AeroComBAT.Structures.Mesher
   :members: boxBeam, laminate, rectBoxBeam

CROSS-SECTION
+++++++++++++

.. autoclass:: AeroComBAT.Structures.XSect
   :members: __init__, xSectionAnalysis, resetResults, calcWarpEffects, printSummary, plotRigid, plotWarped

TIMOSHENKO BEAM
+++++++++++++++

.. autoclass:: AeroComBAT.Structures.TBeam
   :members: __init__, printSummary, plotRigidBeam, plotDisplBeam, printInternalForce

SUPER-BEAM
++++++++++

.. autoclass:: AeroComBAT.Structures.SuperBeam
   :members: __init__, getBeamCoord, printInternalForce, writeDisplacements, writeForcesMoments, getEIDatx, printSummary

WING SECTION
++++++++++++

.. autoclass:: AeroComBAT.Structures.WingSection
   :members: __init__, plotRigid, plotDispl