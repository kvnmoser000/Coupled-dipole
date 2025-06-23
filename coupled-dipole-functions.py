import numpy as np

class Dipole:
    def __init__(self, position, orientation, alpha_id):
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)  # 3-vector or 3x3 rotation matrix
        self.alpha_id = alpha_id  # integer key into the polarizability dictionary

class DipoleSystem:
    def __init__(self, dipoles, polarizability_dict):
        self.dipoles = dipoles
        self.N = len(dipoles)
        self.polarizability_dict = polarizability_dict