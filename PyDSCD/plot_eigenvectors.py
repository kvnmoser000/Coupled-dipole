import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

hbar = 6.582119569e-16

data = np.load("dipole_data.npz")

eigenvalues = data["eigenvalues"]
eigenvectors = data["eigenvectors"]
positions = np.array(data["positions"])*1e7
orientations = np.array(data["orientations"])

unique_positions, inverse_indices = np.unique(positions, axis=0, return_inverse=True)
collapsed_eigmode = np.zeros((len(unique_positions), 3))

k = 8

coeffs = eigenvectors[:, k] ### grab each eigenvector
eigmode = (coeffs*orientations.T).T
eigval = eigenvalues[k]

energy = np.round(np.real(np.sqrt(eigval) * hbar), 2)

for i in range(len(positions)):
    idx=inverse_indices[i]
    collapsed_eigmode[idx]+=eigmode[i]
def do_3d_projection(self, renderer=None):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    return np.min(zs)  # or return 0


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scale = 19  # scale down arrows for display
for i in range(len(unique_positions)):
    start = unique_positions[i]
    vec = collapsed_eigmode[i]

    print(np.shape(start))
    end = start + vec * scale

    arrow = Arrow3D([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    mutation_scale=10,
                    lw=2,
                    arrowstyle="-|>",
                    color="darkred",
                    alpha=0.8)
    ax.add_artist(arrow)

ax.scatter(unique_positions[:, 0],
           unique_positions[:, 1],
           unique_positions[:, 2],
           color='k', s=5,alpha = 1)

ax.set_box_aspect([1, 1, 1])
ax.set_title(f"Eigenmode {k} — Energy = {energy} eV")
plt.tight_layout()
#ax.view_init(elev=90, azim=-90)
plt.xlabel('x (nm)')
#ax.set_axis_off()

plt.show()

