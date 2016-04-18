=======================
AeroComBAT Introduction
=======================

AeroComBAT (Aeroelastic Composite Beam Analysis Tool) is a python API intended
to allow users to efficiently models composite beam structures.

:Authors: 
    Ben Names

:Version: 1.0 of 2016/04/16

Version 1.0 
===========

**Capabilities**

- Simple classical lamination theory analysis
- Cross-sectional analysis of composite beams
- 3D Composite Timoshenko (shear deformable) beam analysis
   + Linear static analysis
   + Normal mode analysis
   + Dynamic Aeroelastic Stability (Flutter) Analysis

Installation Instructions
=========================

First of all it is strongly recomended that the user first install the Anaconda
python distribution from Continuum analytics `here <https://www.continuum.io/>`_.

By installing Anaconda, you will automatically get 3 of the AeroComBAT
dependencies, Numpy, Scipy, and Numba. The last dependency is the visualizer
MayaVi. The easiest way to install this is with Anaconda's package management
'conda'. To install this, simply run:

.. code-block:: python

   conda install mayavi

Mayavi is the 3D visualization engine currently used by AeroComBAT. Note that
in some cases installing mayavi has been found to downgrade numpy. This is not
necessary, so try to update numpy after installing MayaVi by executing:

.. code-block:: python

   conda update numpy

Documentation and tutorials
===========================

For for the AeroComBAT documentation and Tutorials,
see `AeroComBAT Tutorials <http://aerocombat-project.readthedocs.org/>`_.