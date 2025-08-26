import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation

hbar = 6.582119569e-16  # eV·s
c = 3e10  # cm/s

# Physical constants
esq = (4.8e-10) ** 2  # (esu)^2
m_eff = 1e-31  # g

# lattice constants
d = 6.84e-7  # cm
a1 = d * np.array([1, 0, 0])
a2 = d * np.array([0.5, np.sqrt(3) / 2, 0])
a3 = np.cross(a1, a2)

# reciprocal lattice
volume = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / volume
b2 = 2 * np.pi * np.cross(a3, a1) / volume


K = 0.1*b2


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

N_max = 20
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
eigvals, eigvecs = eigh(A_mat, B_mat)

###################### Plotting Fields ##########################################

n = 5
w = np.real(np.sqrt(np.abs(eigvals[n])))  # energy in eV
eigvecs_split = eigvecs.T.reshape(-1, 2, 3)
p_A = eigvecs_split[n, 0]  # 3-vector for site A
p_B = eigvecs_split[n, 1]  # 3-vector for site B

N_max = 5

n_vals = np.arange(-N_max, N_max + 1)
m_vals = np.arange(-N_max, N_max + 1)
n_grid, m_grid = np.meshgrid(n_vals, m_vals, indexing='ij')
R_vectors = n_grid[..., None] * a1 + m_grid[..., None] * a2  # shape (2N+1, 2N+1, 3)

R_flat = R_vectors.reshape(-1, 3)
R_A = R_flat + rA
R_B = R_flat + rB

phases_A = np.exp(1j * np.dot(R_A, K))

phases_A = np.exp(1j * (R_A @ K))
phases_B = np.exp(1j * np.dot(R_B, K))
phases_B = np.exp(1j * (R_B @ K))


dipoles_A = p_A[None, :] * phases_A[:, None]
dipoles_B = p_B[None, :] * phases_B[:, None]


ngrid = 100
y = np.linspace(-4*d, 4*d, ngrid)
z = np.linspace(-1*d, 2*d, ngrid)
Y, Z = np.meshgrid(y, z)
X = np.full_like(Y, 0.25 * d)
obs_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N_points,3)

def calc_field_all(dip_positions, dip_vectors, obs_points, w):
    """
    Fully vectorized: return E at each obs point due to all dipoles.
    dip_positions: (Nd,3)
    dip_vectors: (Nd,3)
    obs_points: (Np,3)
    returns E: (Np,3)
    """
    k = w / c
    # geometry arrays
    r = obs_points[:, None, :] - dip_positions[None, :, :]       # (Np, Nd, 3)
    r_mag = np.linalg.norm(r, axis=2)                            # (Np, Nd)
    # avoid divide-by-zero (move self-contributions far from observation plane)
    tiny = 1e-12
    r_mag_safe = np.where(r_mag < tiny, tiny, r_mag)
    n = r / r_mag_safe[:, :, None]                               # (Np, Nd, 3)
    p = dip_vectors[None, :, :]                                  # (1, Nd, 3)

    # term1: k^2 (n x p) x n * e^{ikr}/r
    n_cross_p = np.cross(n, p)                                   # (Np,Nd,3)
    term1 = np.cross(n_cross_p, n) * (k ** 2 * np.exp(1j * k * r_mag_safe)[:, :, None] / r_mag_safe[:, :, None])

    # term2: [3 n (n·p) - p] * (1/r^3 - i k / r^2) e^{ikr}
    dot_np = np.sum(n * p, axis=2)[:, :, None]                   # (Np,Nd,1)
    pref = (1.0 / r_mag_safe[:, :, None] ** 3) - 1j * k / (r_mag_safe[:, :, None] ** 2)
    term2 = (3 * n * dot_np - p) * pref * np.exp(1j * k * r_mag_safe)[:, :, None]

    E = np.sum(term1 + term2, axis=1)                             # (Np,3)
    return E

dip_positions = np.vstack([R_A, R_B])       # (2*Nd, 3)
dip_vectors   = np.vstack([dipoles_A, dipoles_B])  # (2*Nd, 3)

# Compute total field
E = calc_field_all(dip_positions, dip_vectors, obs_pts, w)
Ex = E[:,0].reshape(ngrid, ngrid)

fig, ax = plt.subplots(figsize=(6, 6))

# Determine symmetric color scale limits so zero maps to white
abs_max = np.max(np.abs(np.real(Ex.reshape(ngrid, ngrid))))
mesh = ax.imshow(np.real(Ex.reshape(ngrid, ngrid)), cmap='bwr',
                 extent = [y.min()*1e7, y.max()*1e7, z.min()*1e7, z.max()*1e7],
                 origin='lower', interpolation='nearest',
                 vmin=-abs_max, vmax=abs_max)

ax.set_xlabel('y (nm)')
ax.set_ylabel('z (nm)')
title = ax.set_title(f"{w*hbar} eV, phase = 0°")

frames = 60
def update(frame):
    phase = 2 * np.pi * frame / frames
    E_phase = Ex * np.exp(1j * phase)
    mesh.set_data(np.real(E_phase.reshape(ngrid, ngrid)))
    title.set_text(f"{np.round(w*hbar,2)} eV")
    return mesh, title

ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Save as GIF using Pillow writer
ani.save('movie.gif', writer='pillow', fps=30, dpi=200)

plt.show()
