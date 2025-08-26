import matplotlib.pyplot as plt
import numpy as np


hbar = 6.582119569e-16
c = 3e10
data = np.load("dipole_data.npz")
eigenvalues = data["eigenvalues"]
eigenvectors = data["eigenvectors"]
positions = np.array(data["positions"])
orientations = np.array(data["orientations"])

N = 200
x = np.linspace(-20e-7, 20e-7, N)
z = np.linspace(-00e-7, 2000e-7, N)
xx, zz = np.meshgrid(x, z)
x_points = xx.ravel()
z_points = zz.ravel()
y_points = np.ones_like(x_points)*10e-7
points = np.stack((x_points, y_points, z_points), axis=-1)
x,y,z = points.T
def calc_field(x_dip,x,y,z,p_vec,w):

    k=w/c
    r_mag=np.sqrt((x_dip[0]-x)**2+(x_dip[1]-y)**2+(x_dip[2]-z)**2)
    r_vec=np.array([x_dip[0]-x,x_dip[1]-y,x_dip[2]-z])
    n=r_vec/r_mag


    B = k**2*np.cross(n,p_vec,axis=0)/(r_mag**2)*(1-1/(1j*k*r_mag))
    E = k**2*np.cross(np.cross(n,p_vec,axis=0),n,axis=0)*(np.exp(1j*k*r_mag)/r_mag)+(
                3*n*np.dot(p_vec,n)-p_vec[:,np.newaxis])*(1/r_mag**3-1j*k/r_mag**2)*np.exp(1j*k*r_mag)
    return E

eval = 3.2      # 2.47 to 2.74        [2.68054 2.54991 2.62381 2.64676 2.56516 2.55861 2.49998 2.67405 2.6237 2.64682 2.56522 2.55855]
window = 0.1
indices = np.where((np.sqrt(eigenvalues) >= (eval-window)/hbar) & (np.sqrt(eigenvalues) <= (eval+window)/hbar))[0]
print(indices)


X = x.reshape(N, N)
Z = z.reshape(N, N)
for k in indices:
    coeffs=eigenvectors[:,k]  ### grab each eigenvector
    eigmode=(coeffs*orientations.T).T
    w=np.sqrt(eigenvalues[k])
    E = np.zeros((3,N**2),dtype = np.complex128)
    for x_dip,p_vec in zip(positions,eigmode):
        E += calc_field(x_dip,x,y,z,p_vec,w)


    chi_E=np.imag(np.cross(E, np.conj(E),axis = 0))
    print(chi_E)

    F=chi_E[2].reshape(N,N)
    fig = plt.figure()
    plt.pcolormesh(X,Z,F,shading='auto',cmap='bwr')
    plt.colorbar(label='Field value')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')



    energy=np.round(np.real(w)*hbar,4)
    plt.title(f"2D Field Plot {energy} eV")
    plt.show()