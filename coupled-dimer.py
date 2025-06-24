import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.linalg import svd
from scipy.signal import argrelextrema
from scipy.linalg import null_space

hbar = 6.582119569e-16
ω_0 = 3/hbar
esq = (4.8e-10)**2
m = 2e-33
esq_m = esq/m
c = 3e10

N = 2 ## number of particles
ω_range = np.linspace(-ω_0*1.4,ω_0*1.4,200)
r = 10e-7  ## 10nm in cm

def alpha(ω):
    return esq_m/(ω_0**2-ω**2)

def g(mu,nu):
    p_hat=np.array([[1,0,0],[0,1,0],[0,0,1]])
    n_hat = np.array([1,0,0])
    return 1/(4*np.pi*r**3)*(3*np.dot(p_hat[mu],n_hat)*np.dot(n_hat,p_hat[nu])-np.dot(p_hat[mu],p_hat[nu]))

r1 = np.array([-0.5,0,0])
r2 = np.array([0.5,0,0])

n_hat = np.array([1,0,0])

p2x_hat = np.array([1,0,0])
p2y_hat = np.array([0,1,0])
p2z_hat = np.array([0,0,1])

detlist = []
alist = []

for ω in ω_range:
    alist.append(alpha(ω))
    amat = np.zeros((3*N,3*N))
    for n in range(3*N):
        for m in range(3*N):
            i,mu=divmod(n,3)  # dipole i and component mu
            j,nu=divmod(m,3)  # dipole j and component nu
            if i == j and mu == nu:
                amat[n,m] = 1/alpha(ω)
            elif i != j:
                amat[n,m] = g(mu,nu)
                print(g(mu,nu))
    det = np.linalg.det(amat)
    detlist.append(det)

fw = np.array(detlist)

minima_indices = argrelextrema(np.abs(fw), np.less)[0]
minima_values = ω_range[minima_indices]*hbar


plt.plot(ω_range*hbar,fw)
plt.scatter(minima_values,np.zeros(np.shape(minima_values)))
plt.show()