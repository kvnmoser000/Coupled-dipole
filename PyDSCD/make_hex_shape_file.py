import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# Lattice parameters
bilayer = False
hexagonal_patch = True
d = 6.84e-7
a1 = d * np.array([1, 0, 0])
a2 = d * np.array([0.5, 0.866, 0])
h = 0*d
theta = 21.79 # degrees
tr_vec = np.array([0,0,0])
Nx, Ny =10,10
R = 15       #9
z_component = True


dipole_data = []

# Constants
E0_eV = 2.60
m_eff = 2.0e-31

def rotate_point(x, y, theta):
    theta_rad = (theta/180)*pi
    x_new = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    y_new = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return x_new, y_new

for i in range(-Nx, Nx):
    for j in range(-Ny, Ny):
        pos = i * a1 + j * a2
        x, y, z = pos

        if hexagonal_patch==False:
            dipole_data.append([x,y,z,1,0,0,E0_eV,m_eff])
            dipole_data.append([x,y,z,0,1,0,E0_eV,m_eff])
            if z_component:
                dipole_data.append([x,y,z,0,0,1,E0_eV,m_eff])

        elif abs(i)+abs(j)+abs(i+j)<=R:
            dipole_data.append([x, y, z, 1, 0, 0, E0_eV, m_eff])
            dipole_data.append([x, y, z, 0, 1, 0, E0_eV, m_eff])
            if z_component:
                dipole_data.append([x, y, z, 0, 0, 1, E0_eV, m_eff])


if bilayer == True:
    for i in range(-Nx,Nx):
        for j in range(-Ny,Ny):
            z_offset = np.array([0,0,1])*h
            pos=i*a1+j*a2-z_offset+tr_vec
            x,y,z=pos

            xnew,ynew = rotate_point(x, y, theta)

            if hexagonal_patch==False:
                dipole_data.append([xnew,ynew,z,1,0,0,E0_eV,m_eff])
                dipole_data.append([xnew,ynew,z,0,1,0,E0_eV,m_eff])
                if z_component:
                    dipole_data.append([xnew,ynew,z,0,0,1,E0_eV,m_eff])
            elif abs(i)+abs(j)+abs(i+j)<=R:
                dipole_data.append([xnew,ynew,z,1,0,0,E0_eV,m_eff])
                dipole_data.append([xnew,ynew,z,0,1,0,E0_eV,m_eff])
                if z_component:
                    dipole_data.append([xnew,ynew,z,0,0,1,E0_eV,m_eff])
# Convert to array
dipole_data = np.array(dipole_data)

# Custom format per column
formats = ['%.2e', '%.2e', '%.2e', '%d', '%d', '%d', '%.2f', '%.1e']
header = "## x\ty\tz\tpx\tpy\tpz\tE0_eV\tm_eff ##"

# Save file
np.savetxt("hexagon_shape.txt",dipole_data,fmt=formats,delimiter="\t",header=header,comments='')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xpts = dipole_data[:,0]
ypts = dipole_data[:,1]
zpts = dipole_data[:,2]
ax.scatter(xpts,ypts,zpts,color='k', s=5,alpha = 1)
ax.set_box_aspect([1, 1, 1])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres.'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    centers = np.mean(limits, axis=1)
    max_range = np.max(np.ptp(limits, axis=1)) / 2

    for ctr, axis in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        axis([ctr - max_range, ctr + max_range])
#set_axes_equal(ax)
ax.view_init(elev=90, azim=0)
plt.show()
print(np.shape(dipole_data))
print(dipole_data)