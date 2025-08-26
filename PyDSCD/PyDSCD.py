import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh

hbar = 6.582119569e-16
c = 3e10
esq = (4.8e-10)**2

########### INPUTS ##########
addr = 'isotropic_dimer_shape.txt'
#############################
def g(rn,rm,pn,pm):
    rmn = np.linalg.norm(rm-rn)
    n_hat = (rn-rm)/rmn
    return -1/(4*np.pi*rmn**3)*(3*np.dot(pm,n_hat)*np.dot(n_hat,pn)-np.dot(pm,pn))

def create_A():
    n=len(dipoles_list)
    A=np.zeros((n,n))
    for n in range(len(dipoles_list)):
        for m in range(len(dipoles_list)):
            dn=dipoles_list[n]
            dm=dipoles_list[m]

            wn,wm=dn.E0_eV/hbar,dm.E0_eV/hbar
            mn,mm=dn.m_eff,dm.m_eff
            rn,rm=dn.position,dm.position
            pn,pm=dn.orientation,dm.orientation

            if (dm.position==dn.position).all() and (dm.orientation==dn.orientation).all():
                A[n,m]=(mm/esq)*(wm)**2
            elif (dm.position!=dn.position).any():
                A[n,m]=g(rn,rm,pn,pm)
    return A

def create_B():
    m_eff_list = []
    for n in range(len(dipoles_list)):
        dn=dipoles_list[n]
        mn=dn.m_eff
        m_eff_list.append(mn)
    return np.diag(np.array(m_eff_list))/esq

class Dipole: ## defines the dipole class which stores the properties of each dipole
    def __init__(self, position, orientation, E0_eV,m_eff):
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)  # 3-vector or 3x3 rotation matrix
        self.E0_eV = E0_eV  # integer key into the polarizability dictionary
        self.m_eff = m_eff  # effective mass (oscillator strength)

dipoles = np.loadtxt(addr, comments='##') ## reads in dipole shape file
dipoles_list = []

for dipole in dipoles: ## creates dipoles_list
    d = Dipole(position = dipole[0:3],orientation = dipole[3:6],E0_eV = dipole[6],m_eff = dipole[7])
    dipoles_list.append(d)
A = create_A()
B = create_B()

eigenvalues, eigenvectors = eig(A, B)
positions = [dipole.position for dipole in dipoles_list]
orientations = [dipole.orientation for dipole in dipoles_list]
np.savez("dipole_data.npz", eigenvalues=eigenvalues, eigenvectors=eigenvectors,\
         positions=positions, orientations=orientations)

energies = np.round(np.real(np.sqrt(eigenvalues) * hbar), 5)
eigvecs = np.round(eigenvectors, 3)



print(f"{'Mode':<5} {'Energy (eV)':<12} Eigenvector (rounded)")
print("-" * 50)

for k in range(len(energies)):
    energy_str = f"{energies[k]:<12}"
    vec_str = np.array2string(eigvecs[:, k], separator=', ', precision=3, suppress_small=True)
    print(f"{k:<5} {energy_str} {vec_str}")

print("Energies:")
print(energies)


