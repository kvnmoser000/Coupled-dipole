import numpy as np
import csv
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.linalg import null_space
from scipy.linalg import svd

hbar = 6.582119569e-16
c = 3e10
esq = (4.8e-10)**2

########### INPUTS ##########
eV_min = 3*0.2
eV_max = 3*1.38
num_points = 2000
#############################


def alpha(ω,ω_0,m_eff):
    esq_m=esq/m_eff
    return esq_m/(ω_0**2-ω**2)
def g(rn,rm,pn,pm):
    rmn = np.linalg.norm(rm-rn)
    n_hat = np.array([0,1,0])
    return 1/(4*np.pi*rmn**3)*(3*np.dot(pm,n_hat)*np.dot(n_hat,pn)-np.dot(pm,pn))
class Dipole: ## defines the dipole class which stores the properties of each dipole
    def __init__(self, position, orientation, E0_eV,m_eff):
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)  # 3-vector or 3x3 rotation matrix
        self.E0_eV = E0_eV  # integer key into the polarizability dictionary
        self.m_eff = m_eff  # effective mass (oscillator strength)

dipoles = np.loadtxt('isotropic_dimer_shape.txt', comments='##') ## reads in dipole shape file
dipoles_list = []

for dipole in dipoles: ## creates dipoles_list
    d = Dipole(position = dipole[0:3],orientation = dipole[3:6],E0_eV = dipole[6],m_eff = dipole[7])
    dipoles_list.append(d)

n = len(dipoles_list)
A = np.zeros((n, n))

detlist = []
eV_range = np.linspace(eV_min,eV_max,num_points)
for en in eV_range:
    w = en/hbar
    for n in range(len(dipoles_list)):
        for m in range(len(dipoles_list)):
            dn = dipoles_list[n]
            dm = dipoles_list[m]

            wn, wm = dn.E0_eV/hbar, dm.E0_eV/hbar
            mn, mm= dn.m_eff, dm.m_eff
            rn, rm = dn.position, dm.position
            pn, pm = dn.orientation, dm.orientation

            if (dm.position == dn.position).all() and (dm.orientation == dn.orientation).all():
                A[n,m] = 1/alpha(w,wn,mn)
            elif (dm.position!=dn.position).any():
                A[n,m] = g(rn,rm,pn,pm)
    det=np.linalg.det(A)
    detlist.append(det)

fw=np.array(detlist)
minima_indices = argrelextrema(np.abs(fw), np.less)[0]
print(fw[minima_indices])
minima_values = eV_range[minima_indices]
plt.plot(eV_range,fw)
plt.plot(eV_range,np.zeros(len(eV_range)))
plt.plot()
plt.show()

for energy in minima_values:
    w=energy/hbar
    for n in range(len(dipoles_list)):
        for m in range(len(dipoles_list)):
            dn=dipoles_list[n]
            dm=dipoles_list[m]

            wn,wm=dn.E0_eV/hbar,dm.E0_eV/hbar
            mn,mm=dn.m_eff,dm.m_eff
            rn,rm=dn.position,dm.position
            pn,pm=dn.orientation,dm.orientation

            if (dm.position==dn.position).all() and (dm.orientation==dn.orientation).all():
                A[n,m]=1/alpha(w,wn,mn)
            elif (dm.position!=dn.position).any():
                A[n,m]=g(rn,rm,pn,pm)

    U,S,Vh=svd(A)
    threshold=1e16
    null_mask=S<threshold
    null_vectors=Vh[null_mask]
    null_vectors = Vh[-1]
    print(np.round(energy, 3), np.round(null_vectors, 3))