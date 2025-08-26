import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation

# ---------------- Constants & lattice ----------------
hbar = 6.582119569e-16
c = 3e10
esq = (4.8e-10) ** 2
m_eff = 1e-31
d = 6.84e-7

# Bravais (hexagonal)
a1 = np.array([1.0, 0.0, 0.0]) * d
a2 = np.array([0.5, np.sqrt(3) / 2, 0.0]) * d
a3 = np.cross(a1, a2)

volume = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / volume
b2 = 2 * np.pi * np.cross(a3, a1) / volume

# high-symmetry / example K
K_point = 0.5 * b1 + 0.5 * b2   # choose M (or change to other K)
K_point = 0.01 * b1   # choose M (or change to other K)

# ---------------- oscillator on-site matrices ----------------
w0x = 3.0 / hbar
w0y = 3.0 / hbar
w0z = 3.0 / hbar
A2 = (m_eff / esq) * np.diag([w0x ** 2, w0y ** 2, w0z ** 2])
B = (m_eff / esq) * np.eye(3)

# ---------------- lattice sum (static Green dyadic) ----------------
def G_static_vec(Rs):
    """Vectorized static dipole Green dyadic for many R vectors (Rs shape (M,3))."""
    eps = 1e-12
    Rnorm = np.linalg.norm(Rs, axis=1)
    mask = Rnorm > eps
    G = np.zeros((Rs.shape[0], 3, 3), dtype=np.complex128)
    if np.any(mask):
        rh = np.zeros_like(Rs)
        rh[mask] = Rs[mask] / Rnorm[mask, None]
        outer = np.einsum('ni,nj->nij', rh, rh)
        G[mask] = -(3 * outer[mask] - np.eye(3)) / (4 * np.pi * Rnorm[mask][:, None, None] ** 3)
    return G

# choose cutoff for lattice sum (tweak if slow)
Nmax = 30
R_list = []
for n in range(-Nmax, Nmax + 1):
    for m in range(-Nmax, Nmax + 1):
        if n == 0 and m == 0:
            continue
        R_list.append(n * a1 + m * a2)
R_arr = np.array(R_list)  # (M,3)

# compute G tensors for all R
G_tensors = G_static_vec(R_arr)   # (M,3,3)
phases = np.exp(1j * (R_arr @ K_point))  # (M,)
# lattice sum S = sum_R G(R) e^{i K·R}
S = np.tensordot(phases, G_tensors, axes=1)  # (3,3)

# ---------------- assemble and diagonalize ----------------
A = A2 + S
e_vals, e_vecs = eigh(A, B)
# eigenvalues e_vals correspond to (w^2?) depends on normalization — preserve your original convention
# take positive-frequency root
w_mode_all = np.real(np.sqrt(np.abs(e_vals)))

# print a couple for inspection
print("Mode energies (eV):", w_mode_all*hbar)

# ---------------- choose a mode and build finite tile of dipoles to compute real-space field -----------
mode_index = 2
p_mode = e_vecs[:, mode_index]      # shape (3,)
w_mode = w_mode_all[mode_index]

# finite tiling for real-space field (this is NOT the infinite-sum eigenproblem; it's a finite-array field evaluation)
Nx = Ny = 7   # increase for larger finite patch (cost grows ~N^2)
positions = np.array([i * a1 + j * a2 for i in range(-Nx, Nx) for j in range(-Ny, Ny)])  # (N_dipoles,3)

# Bloch-phase each lattice site so the finite tile approximates the Bloch mode:
phases_tile = np.exp(1j * (positions @ K_point))   # (N_dipoles,)
dipoles = p_mode[None, :] * phases_tile[:, None]    # (N_dipoles,3)

# ---------------- observation grid (sheet d x d elevated 0.1*d) ----------------
ngrid = 200
x = np.linspace(-4*d, 4*d, ngrid)
y = np.linspace(-4*d, 4*d, ngrid)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 0.25 * d)
obs_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N_points,3)

# ---------------- vectorized field calculator using retarded dipole formula (E-field) -----------
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

E = calc_field_all(positions, dipoles, obs_pts, w_mode)
Ez = E[:,2].reshape(ngrid, ngrid)


fig, ax = plt.subplots(figsize=(6, 6))

# Determine symmetric color scale limits so zero maps to white
abs_max = np.max(np.abs(np.real(Ez.reshape(ngrid, ngrid))))
mesh = ax.imshow(np.real(Ez.reshape(ngrid, ngrid)), cmap='bwr',
                 extent = [x.min()*1e7, x.max()*1e7, y.min()*1e7, y.max()*1e7],
                 origin='lower', interpolation='nearest',
                 vmin=-abs_max, vmax=abs_max)

ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
title = ax.set_title(f"{w_mode*hbar} eV, phase = 0°")

frames = 60
def update(frame):
    phase = 2 * np.pi * frame / frames
    E_phase = Ez * np.exp(1j * phase)
    mesh.set_data(np.real(E_phase.reshape(ngrid, ngrid)))
    title.set_text(f"{w_mode*hbar} eV")
    return mesh, title

ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Save as GIF using Pillow writer
ani.save('movie.gif', writer='pillow', fps=30, dpi=200)

plt.show()




