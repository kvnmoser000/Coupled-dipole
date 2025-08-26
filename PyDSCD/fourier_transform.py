import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.interpolate import griddata
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d

hbar = 6.582119569e-16
c = 3e10
d = 6.84e-7

data = np.load("dipole_data.npz")
eigenvalues = data["eigenvalues"]
eigenvectors = data["eigenvectors"]
positions = np.array(data["positions"])
orientations = np.array(data["orientations"])

a1 = d * np.array([1, 0])
a2 = d * np.array([0.5, 0.866])
A = np.column_stack((a1, a2))
A_inv = np.linalg.inv(A)


klist = []
elist = []

i = 100
coeffs=eigenvectors[:,i]  ### grab each eigenvector
eigmode=(coeffs*orientations.T).T
w=np.sqrt(eigenvalues[i])

x,y=positions[:,0],positions[:,1]
px,py=eigmode[:,0],eigmode[:,1]

points = np.stack((x, y), axis = 1)
dips = np.stack((px,py), axis = 1)

indices = []
for r in points:
    mn = A_inv @ r
    m, n = np.round(mn)
    indices.append((m, n))

indices = np.array(indices)

m_values = indices[:, 0]
n_values = indices[:, 1]

unique_m = np.unique(m_values)
unique_n = np.unique(n_values)

d = 6.84e-7  # spacing in cm

# Store FFT magnitudes and k-values
fft_m_spectra = []
fft_n_spectra = []

# --- FFT along constant-m lines (varying n direction) ---
for m in unique_m:
    mask = m_values == m
    x_filtered = positions[mask, 0]
    y_filtered = positions[mask, 1]
    px_filtered = dips[mask, 0]

    # Sort by y (or project along a2 direction)
    sort_idx = np.argsort(y_filtered)
    px_sorted = px_filtered[sort_idx]

    N = len(px_sorted)
    if N < 4:  # skip short lines
        continue

    fft_px = fft(px_sorted)
    k_vals = fftfreq(N, d)  # units of cm⁻¹

    fft_m_spectra.append((fftshift(k_vals), fftshift(np.abs(fft_px))))

# --- FFT along constant-n lines (varying m direction) ---
for n in unique_n:
    mask = n_values == n
    x_filtered = positions[mask, 0]
    y_filtered = positions[mask, 1]
    px_filtered = dips[mask, 0]

    # Sort by x (or project along a1 direction)
    sort_idx = np.argsort(x_filtered)
    px_sorted = px_filtered[sort_idx]

    N = len(px_sorted)
    if N < 4:
        continue

    fft_px = fft(px_sorted)
    k_vals = fftfreq(N, d)

    fft_n_spectra.append((fftshift(k_vals), fftshift(np.abs(fft_px))))




all_k_vals = np.concatenate([spec[0] for spec in fft_m_spectra] + [spec[0] for spec in fft_n_spectra])
k_min, k_max = all_k_vals.min(), all_k_vals.max()

# Define a uniform common k-grid with enough points
num_k_points = 512
common_k = np.linspace(k_min, k_max, num_k_points)

def interpolate_spectra(fft_spectra):
    interp_specs = []
    for k_vals, amplitudes in fft_spectra:
        interp_func = interp1d(k_vals, amplitudes, bounds_error=False, fill_value=0)
        interp_amplitudes = interp_func(common_k)
        interp_specs.append(interp_amplitudes)
    return np.array(interp_specs)

# Interpolate all spectra to the common k-grid
interp_fft_m = interpolate_spectra(fft_m_spectra)
interp_fft_n = interpolate_spectra(fft_n_spectra)

# Average along the lines (axis=0)
avg_m_spectrum = np.mean(interp_fft_m, axis=0)
avg_n_spectrum = np.mean(interp_fft_n, axis=0)

# Plot the averages
plt.figure(figsize=(8, 5))
plt.plot(common_k, avg_m_spectrum, label='Avg FFT |px| (constant m)', color='blue')
plt.plot(common_k, avg_n_spectrum, label='Avg FFT |px| (constant n)', color='red', linestyle='--')
plt.xlabel('k (cm⁻¹)')
plt.ylabel('Average FFT Amplitude')
plt.title('Average FFT Spectra along Constant m and n Lines')
plt.legend()
plt.grid(True)
plt.show()



