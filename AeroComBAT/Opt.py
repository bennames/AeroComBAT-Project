# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================
from AircraftParts import Wing

# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np

# =============================================================================
# WING OPTIMIZATION CLASS
# =============================================================================
class WingOpt:
    def __init__(self):
        self.cons = []
        self.obj = []
    def instantiate(self,b_s,croot,ctip,x0_spar,x1_spar,Y_rib,n_ply,m_i,g,mass,mat_lib):
        self.wing = Wing(b_s,croot,ctip,x0_spar,x1_spar,Y_rib,n_ply,m_i,g,mass,mat_lib)
        