import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

hbar = 6.582119569e-16  # eV·s
c = 3e10  # cm/s

# Physical constants
esq = (4.8e-10) ** 2  # (esu)^2
m_eff = 1e-31  # g

# lattice constants
d = 6.84e-7  # cm
a1 = d * np.array([1, 0, 0])
a2 = d * np.array([0.5, np.sqrt(3) / 2, 0])

a1 = d * np.array([1, 0, 0])
a2 = d * np.array([0.5, np.sqrt(3) / 2, 0])

a3 = np.cross(a1, a2)

# reciprocal lattice
volume = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / volume
b2 = 2 * np.pi * np.cross(a3, a1) / volume

# path in reciprocal space
Gamma = np.array([0.0, 0.0, 0.0])
K = (1 / 3) * b1 + (2 / 3) * b2
M = 0.5 * b1 + 0.5 * b2
N = 150

path_MG = np.linspace(M, Gamma, N, endpoint=False)
path_GK = np.linspace(Gamma, K, N, endpoint=False)
path_KM = np.linspace(K, M, N, endpoint=False)
k_path = np.vstack([path_MG, path_GK, path_KM])

# oscillator parameters
w0x = 3.0 / hbar
w0y = 3.0 / hbar
w0z = 3.0 / hbar
A0 = (m_eff / esq) * np.diag([w0x ** 2, w0y ** 2, w0z ** 2])  # 3x3
B0 = (m_eff / esq) * np.eye(3)  # 3x3

# basis positions
#rA = np.array([0, 0, 0])
#rB = (a1 + a2) / 3

dsep = 1

rA = np.array([0, 0, 0])
rB = dsep*d*np.array([0, 0, 1])

basis = np.array([rA, rB])  # shape (2,3)

N_max = 40
eigeVs = []

# Prepare lattice vectors R = n a1 + m a2, excluding n=m=0 for self-terms per basis-site later
n_vals = np.arange(-N_max, N_max + 1)
m_vals = np.arange(-N_max, N_max + 1)
n_grid, m_grid = np.meshgrid(n_vals, m_vals, indexing='ij')
R_vectors = n_grid[..., None] * a1 + m_grid[..., None] * a2  # shape (2N+1, 2N+1, 3)

# Flatten lattice vectors for vectorized calculations
R_flat = R_vectors.reshape(-1, 3)  # (M, 3) with M = (2N+1)^2

def G0_vec(r_vecs):
    """Vectorized G0 function for multiple vectors r_vecs (shape (M,3))"""
    r_norms = np.linalg.norm(r_vecs, axis=1)  # (M,)
    mask = r_norms > 1e-15  # avoid division by zero
    G = np.zeros((r_vecs.shape[0], 3, 3), dtype=np.complex128)
    r_hat = np.zeros_like(r_vecs)
    r_hat[mask] = r_vecs[mask] / r_norms[mask, None]
    outer = np.einsum('ni,nj->nij', r_hat, r_hat)
    G[mask] = -(3 * outer[mask] - np.eye(3)) / (4 * np.pi * r_norms[mask][:, None, None] ** 3)
    return G

for K in k_path:
    # Initialize 2x2 blocks of 3x3 zero matrices
    S_blocks = np.zeros((2, 2, 3, 3), dtype=np.complex128)

    # Compute S_blocks[i,j]
    for i in range(2):
        for j in range(2):
            # Calculate all relative position vectors r = R + r_j - r_i
            r_ij = basis[j] - basis[i]  # shape (3,)
            r_all = R_flat + r_ij  # shape (M,3)

            # Identify self-term for i==j at R=0, exclude it
            if i == j:
                mask_self = np.all(np.isclose(r_all, 0), axis=1)
            else:
                mask_self = np.zeros(r_all.shape[0], dtype=bool)

            # Compute Green's tensors for all vectors except self-term
            G_all = G0_vec(r_all)

            # Phase factors
            phases = np.exp(1j * np.dot(r_all, K))

            # Sum over lattice vectors excluding self term
            S_blocks[i, j] = np.sum(G_all[~mask_self] * phases[~mask_self, None, None], axis=0)

    # Assemble full 6x6 matrices A and B
    A_mat = np.zeros((6, 6), dtype=np.complex128)
    B_mat = np.zeros((6, 6), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            row = slice(3 * i, 3 * (i + 1))
            col = slice(3 * j, 3 * (j + 1))
            A_mat[row, col] = S_blocks[i, j]
            if i == j:
                A_mat[row, col] += A0
                B_mat[row, col] = B0

    # Solve generalized eigenvalue problem
    eigvals, _ = eigh(A_mat, B_mat)
    eigeVs.append(np.real(np.sqrt(np.abs(eigvals))) * hbar)  # energy in eV

eigeVs = np.array(eigeVs)

plt.figure(figsize=(6, 4))
for i in range(eigeVs.shape[1]):
    plt.plot(eigeVs[:, i], 'k-', markersize=1)

xticks = [0, N, 2 * N, 3 * N]
xtick_labels = [r'$M$', r'$\Gamma$', r'$K$', r'$M$']
plt.xticks(xticks, xtick_labels)
plt.ylabel('Energy (eV)')
plt.title('')
plt.tight_layout()
plt.plot([], [], ' ', label=f"Separation = {str(np.round(d*dsep*1e7))} nm")
plt.legend(loc="upper right")
plt.show()

