import numpy as np
import matplotlib.pyplot as plt

E0_eV = 3.0

Ndips = 200
L = 2000e-7     ### lenght in nm
d = L/Ndips
m_eff = (1*10**(-44))/d**2
L = Ndips*d
print(L)
tw_per_dip = 0.1    #### 0.1
Ntwists = tw_per_dip*Ndips
print(Ntwists)
hand = -1

p0 = np.array([1,0,0])
x0 = np.array([0,0,0])
dipole_data = []

def rotate_z(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

dtheta = 2*np.pi*Ntwists/Ndips
dx = d*np.array([0,0,1])
for i in range(Ndips):
    pos = rotate_z(hand*i*dtheta)@(x0+i*dx)
    dip = rotate_z(hand*i*dtheta)@p0

    dipole_data.append([pos[0],pos[1],pos[2],dip[0],dip[1],dip[2],E0_eV,m_eff])

dipole_data = np.array(dipole_data)

formats = ['%.2e', '%.2e', '%.2e', '%.2e', '%.2e', '%.2e', '%.2f', '%.1e']
header = "## x\ty\tz\tpx\tpy\tpz\tE0_eV\tm_eff ##"
np.savetxt("helix_shape.txt",dipole_data,fmt=formats,delimiter="\t",header=header,comments='')