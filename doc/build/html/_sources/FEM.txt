FEM Interface Module
--------------------

.. automodule:: AeroComBAT.FEM

MODEL
+++++
.. autoclass:: AeroComBAT.FEM.Model
   :members: addElements, addAircraftParts, resetPointLoads, resetResults,
             applyLoads, applyConstraints, staticAnalysis, normalModesAnalysis,
             flutterAnalysis, plotRigidModel, plotDeformedModel
LOAD SET
++++++++
.. autoclass:: AeroComBAT.FEM.LoadSet
   :members: addPointLoad, addDistributedLoad

FLUTTER POINT
+++++++++++++
.. autoclass:: AeroComBAT.FEM.FlutterPoint
   :members: __init__, saveSol, interpOmegaRoot

