import numpy as np
import matplotlib.pyplot as plt


nearest_neighbor = True

m = 1
k = 1
a = 1

def alpha_inv(w):
    k-m*w**2





def wsq(q,num_neighbors = 1):

    p=0.25

    if num_neighbors==1:
        n_vals=np.array([-1,1])
    if num_neighbors==2:
        nn_vals=np.array([-2,2])
        n_vals=np.array([-1,1])
        exp_term_nn=np.exp(1j*np.outer(nn_vals,q)*a)  # shape: (len(n_vals), len(q))

    q = np.atleast_1d(q)  # ensures q is an array
    exp_term = np.exp(1j * np.outer(n_vals, q) * a)  # shape: (len(n_vals), len(q))

    if num_neighbors == 1:
        mwsq = -k * np.sum(exp_term, axis=0) + 2 * k
    if num_neighbors == 2:
        mwsq=-k*np.sum(exp_term,axis=0)-p*k*np.sum(exp_term_nn,axis=0)+2*k+p*2*k
    return np.real(mwsq / m)


qrange = np.linspace(-np.pi/a,np.pi/a,200)

plt.plot(qrange,np.sqrt(wsq(qrange,1)))
#plt.plot(qrange,np.sqrt(wsq(qrange,2)))
plt.show()