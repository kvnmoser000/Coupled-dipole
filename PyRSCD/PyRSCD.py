import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


hbar = 6.582119569e-16
c = 3e10

########## User Inputs ##########

# Physical constants
esq             = (4.8e-10)**2         # e² in statcoulombs²
m_eff           = 1e-31                # Effective mass [g]
d               = 5e-7                # Lattice spacing [cm]

# Lattice settings
bravais_lattice = 'Square'            # Options: 'Square', 'Hexagonal'

# Brillouin zone path
N = 150                              # Number of steps in BZ
path_string     = 'GM'              # High-symmetry path, e.g. 'XGMX', 'MGKM', 'Custom'

#################################

if bravais_lattice == 'Square':
    a1 = d * np.array([1, 0, 0])
    a2 = d * np.array([0, 1, 0])
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
    k_path = []

    def Rz(theta):
        """Rotation matrix about the z-axis by angle theta (in radians)."""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
    num_thetas = 9
    thetas = np.linspace(0,np.pi/4, num_thetas)  # 4 angles from 0° to 90°
    start = np.array([0, 0, 0])

    for theta in thetas:
        end = Rz(theta) @ ((2 * np.pi / (0.6*d)) * np.array([1/np.sqrt(2), 1/np.sqrt(2), 0]))
        segment = np.linspace(start, end, N, endpoint=False)
        k_path.append(segment)

    k_path = np.vstack(k_path)  # Combine segments into one continuous array


# Eigenvalue storage
eigeVs = []

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

plt.figure(figsize=(6, 4))

if path_string == 'Custom':
    for segment in np.array_split(eigeVs,num_thetas):
        plt.plot(segment,color = 'k')
else:
    plt.plot(eigeVs, color='k')
    #xticks = [0, N, 2*N, 3*N]
    #xtick_labels = [r'$M$',r'$\Gamma$',r'$K$', r'$M$']  ##### automatically label (fix)
    #plt.xticks(xticks, xtick_labels)
    plt.ylabel('Energy (eV)')
plt.show()


