import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.interpolate import interp1d


hbar = 6.582119569e-16
c = 3e10

########## User Inputs ##########

# Physical constants
esq             = (4.8e-10)**2         # e² in statcoulombs²
m_eff           = 1e-31                # Effective mass [g]
d               = 10e-7                # Lattice spacing [cm]

# Lattice settings
bravais_lattice = 'Square'            # Options: 'Square', 'Hexagonal'

# Brillouin zone path
N = 150                              # Number of steps in BZ
path_string     = 'Custom'              # High-symmetry path, e.g. 'XGMX', 'MGKM', 'Custom'

#################################

if bravais_lattice == 'Square':
    a1 = d * np.array([0.5, 0, 0])
    a2 = d * np.array([0, 0.7, 0])
elif bravais_lattice == 'Hexagonal':
    a1 = d * np.array([1, 0, 0])
    a2 = d * np.array([0.5, np.sqrt(3)/2, 0])
else:
    raise ValueError(f'Unknown bravais_lattice: {bravais_lattice}')

a3=np.cross(a1,a2)


volume = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / volume
b2 = 2 * np.pi * np.cross(a3, a1) / volume



if bravais_lattice == 'Square':
    high_sym_pts = {
        'G': np.array([0.0, 0.0, 0.0]),
        'X': 0.5 * b2,
        'X2':0.5*b1,
        'M': 0.5 * b1 + 0.5 * b2,
    }
elif bravais_lattice == 'Hexagonal':
    high_sym_pts = {
        'G': np.array([0.0, 0.0, 0.0]),
        'M': 0.5 * b1 + 0.5 * b2,
        'K': (1/3) * b1 + (2/3) * b2,
    }

invalid_pts=[pt for pt in path_string if pt not in high_sym_pts]
if invalid_pts and path_string != 'Custom':
    raise ValueError(f"Invalid point(s) in path_string for {bravais_lattice}: {invalid_pts}")

if path_string != 'Custom':
    k_path = []

    for i in range(len(path_string) - 1):
        start = high_sym_pts[path_string[i]]
        end = high_sym_pts[path_string[i + 1]]
        segment = np.linspace(start, end, N, endpoint=False)
        k_path.append(segment)

    # Append the final point
    k_path.append(np.array([high_sym_pts[path_string[-1]]]))

    # Stack into one array
    k_path=np.vstack(k_path)

if path_string == 'Custom':
    k_path=[]
    n_rays=5  # rays from Γ → X to M (includes M)

    # Generate edge points
    vertical_edge=np.linspace(high_sym_pts['X'],high_sym_pts["M"],n_rays)
    horizontal_edge=np.linspace(high_sym_pts['M'],high_sym_pts["X2"],n_rays)[1:]
    # [1:] skips M to avoid duplication

    # Rays to vertical edge (X → M)
    for end in vertical_edge:
        segment=np.linspace(high_sym_pts['G'],end,N,endpoint=True)
        k_path.append(segment)

    # Rays to horizontal edge (M → X2, excluding duplicate M)
    for end in horizontal_edge:
        segment=np.linspace(high_sym_pts['G'],end,N,endpoint=True)
        k_path.append(segment)

    k_path=np.vstack(k_path)  # Shape (9*N, 3)
    print(f"{k_path.shape[0]//N} rays generated")  # Should print 9
    q_magnitudes=np.linalg.norm(k_path,axis=1)

# Eigenvalue storage
eigeVs = []
eigvecs = []

# Lattice sum cutoff
N_max = 50
r0 = np.zeros(3)

# Diagonal matrix A2 and mass matrix B
w0x = 3 / hbar
w0y = 3 / hbar
w0z = 3 / hbar
A2 = (m_eff / esq) * np.diag([w0x**2, w0y**2, w0z**2])
B = (m_eff / esq) * np.eye(3)

# Precompute lattice vectors
n_vals = np.arange(-N_max, N_max + 1)
m_vals = np.arange(-N_max, N_max + 1)
n_grid, m_grid = np.meshgrid(n_vals, m_vals, indexing='ij')
mask = ~(np.logical_and(n_grid == 0, m_grid == 0))
n_flat = n_grid[mask]
m_flat = m_grid[mask]
r_vecs = np.outer(n_flat, a1) + np.outer(m_flat, a2)  # shape (M, 3)
M_rnm = r_vecs.shape[0]

# Fully vectorized G0 function (returns shape (M, 3, 3), complex)
def G0_batch(r_vecs):
    r_norms = np.linalg.norm(r_vecs, axis=1)  # shape (M,)
    r_hat = r_vecs / r_norms[:, np.newaxis]   # shape (M, 3)
    outer = np.einsum('ni,nj->nij', r_hat, r_hat)  # shape (M, 3, 3)
    G_real = -(3 * outer - np.eye(3)) / (4 * np.pi * r_norms[:, np.newaxis, np.newaxis]**3)
    return G_real.astype(np.complex128)

# Precompute Green's tensor (real part, independent of K)
G_static = G0_batch(r_vecs)  # shape (M, 3, 3), complex

# Loop over k-points
for K in k_path:
    phase = np.exp(1j * (r_vecs @ K))        # shape (M,), complex
    G_array = G_static * phase[:, np.newaxis, np.newaxis]  # shape (M, 3, 3), complex
    S = np.sum(G_array, axis=0)                 # shape (3, 3), complex
    A = A2 + S
    eigenvalues, eigenvectors = eigh(A, B)
    eigeVs.append(np.real(np.sqrt(eigenvalues)) * hbar)
    eigvecs.append(eigenvectors)


plt.figure(figsize=(6, 4))
flat = np.array(eigeVs).ravel()
plt.hist(flat, bins=100)
plt.xlabel('Energy (eV)')
plt.ylabel('Counts')
plt.show()
q_segments = np.array_split(q_magnitudes, n_rays * 2 - 1)
k_path_segments = np.array_split(k_path, n_rays * 2 - 1)
eig_segments = np.array_split(eigeVs, n_rays * 2 - 1)
eigvec_segments = np.array_split(eigvecs, n_rays * 2 - 1)

plt.figure(figsize=(6, 4))
colors = [
    'black', '#800020', 'red', 'orange', 'gold',
    'green', 'blue', 'purple', 'lightblue'
]

for i, (q_seg, eig_seg, eigvec_seg, k_path_seg) in enumerate(
    zip(q_segments, eig_segments, eigvec_segments, k_path_segments)
):
    color = colors[i % len(colors)]  # cycle colors
    qz_norm = np.array([0.0, 0.0, 1.0], dtype=float)  # out-of-plane direction

    # Reference in-plane k direction (fixed per segment)
    k_vec = k_path_seg[-1] / np.linalg.norm(k_path_seg[-1])
    k_norm = np.array([k_vec[0], k_vec[1], 0.0], dtype=float)
    t_norm = np.cross(qz_norm, k_norm)

    # Arrays to collect tracked modes
    oop_vals, long_vals, trans_vals = [], [], []

    # Initialize mode assignment at the LAST point (high q)
    q_last = q_seg[-1]
    eigvals_last = eig_seg[-1]
    eigvecs_last = eigvec_seg[-1]

    # Classify just once at the high-q end
    proj_qz = [np.abs(np.dot(eigvecs_last[:, j], qz_norm)) for j in range(len(eigvals_last))]
    oop_idx = np.argmax(proj_qz)

    proj_k = [np.abs(np.dot(eigvecs_last[:, j], k_norm)) if j != oop_idx else -1 for j in range(len(eigvals_last))]
    long_idx = np.argmax(proj_k)

    trans_idx = [j for j in range(len(eigvals_last)) if j not in (oop_idx, long_idx)][0]

    # Save first values
    oop_vals.append((q_last, eigvals_last[oop_idx]))
    long_vals.append((q_last, eigvals_last[long_idx]))
    trans_vals.append((q_last, eigvals_last[trans_idx]))

    # Store eigenvectors for tracking
    prev_vecs = {
        "oop": eigvecs_last[:, oop_idx],
        "long": eigvecs_last[:, long_idx],
        "trans": eigvecs_last[:, trans_idx],
    }

    # Walk backwards through q (high -> low)
    for q_val, eigvals, eigvecs_k in zip(q_seg[::-1][1:], eig_seg[::-1][1:], eigvec_seg[::-1][1:]):
        # For each mode, find best overlap with previous vector
        overlaps = np.abs(eigvecs_k.T.conj() @ prev_vecs["oop"])
        oop_idx = np.argmax(overlaps)
        oop_vals.append((q_val, eigvals[oop_idx]))

        overlaps = np.abs(eigvecs_k.T.conj() @ prev_vecs["long"])
        long_idx = np.argmax(overlaps)
        long_vals.append((q_val, eigvals[long_idx]))

        overlaps = np.abs(eigvecs_k.T.conj() @ prev_vecs["trans"])
        trans_idx = np.argmax(overlaps)
        trans_vals.append((q_val, eigvals[trans_idx]))

        # Update reference eigenvectors
        prev_vecs = {
            "oop": eigvecs_k[:, oop_idx],
            "long": eigvecs_k[:, long_idx],
            "trans": eigvecs_k[:, trans_idx],
        }

    # Convert to arrays for plotting
    oop_vals   = np.array(oop_vals)
    long_vals  = np.array(long_vals)
    trans_vals = np.array(trans_vals)

    plt.plot(oop_vals[:,0], oop_vals[:,1], color=color, linestyle='solid')
    plt.plot(long_vals[:,0], long_vals[:,1], color=color, linestyle='dashed')
    plt.plot(trans_vals[:,0], trans_vals[:,1], color=color, linestyle='dotted')

plt.plot([], [], color='black', linestyle='solid',  label='Out-of-plane (OOP)')
plt.plot([], [], color='black', linestyle='dashed', label='Longitudinal')
plt.plot([], [], color='black', linestyle='dotted', label='Transverse')
plt.xlabel(r'$|q|$ (cm$^{-1}$)')
plt.ylabel('Energy (eV)')
plt.legend(loc='best', frameon=False)
plt.show()



