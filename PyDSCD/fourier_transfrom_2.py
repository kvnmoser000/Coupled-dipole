import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.interpolate import griddata
from scipy.fft import fft, fftfreq, fftshift,fft2
from scipy.interpolate import interp1d

hbar = 6.582119569e-16
c = 3e10
d = 6.84e-7

data = np.load("dipole_data.npz")
eigenvalues = data["eigenvalues"]
eigenvectors = data["eigenvectors"]
positions = np.array(data["positions"])
orientations = np.array(data["orientations"])
klist = []

for i in range(len(eigenvalues)):
    print(i)
    coeffs=eigenvectors[:,i]  ### grab each eigenvector
    eigmode=(coeffs*orientations.T).T
    w=np.sqrt(eigenvalues[i])

    x,y=positions[:,0],positions[:,1]
    px,py=eigmode[:,0],eigmode[:,1]
    p = np.sqrt(px**2+py**2)-np.mean(np.sqrt(px**2+py**2))

    if i == 0:
        num_grid_points = 256
        x_lin = np.linspace(x.min(), x.max(), num_grid_points)
        y_lin = np.linspace(y.min(), y.max(), num_grid_points)
        X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

        # Frequency axes
        kx=fftshift(fftfreq(num_grid_points,d))*2*np.pi
        ky=fftshift(fftfreq(num_grid_points,d))*2*np.pi
        KX,KY=np.meshgrid(kx,ky,indexing='ij')

    p_grid=griddata(points=(x,y),values=p,xi=(X_grid,Y_grid),method='linear',fill_value=0)

    # Zero-pad p_grid before FFT
    pad_factor=4
    pad_size=int((pad_factor-1)*num_grid_points/2)
    p_grid_padded=np.pad(p_grid,pad_size,mode='constant',constant_values=0)

    # 2D FFT on padded data
    p_fft=fftshift(fft2(p_grid_padded))
    p_fft_amp=np.abs(p_fft)

    # Updated frequency axes for padded grid
    kx=fftshift(fftfreq(p_grid_padded.shape[0],d))*2*np.pi
    ky=fftshift(fftfreq(p_grid_padded.shape[1],d))*2*np.pi
    KX,KY=np.meshgrid(kx,ky,indexing='ij')

    # Find peak in FFT amplitude
    max_idx=np.unravel_index(np.argmax(p_fft_amp),p_fft_amp.shape)
    kx_max=KX[max_idx]
    ky_max=KY[max_idx]

    # Compute k
    k=np.sqrt(kx_max**2+ky_max**2)
    #k = ky_max  # use this instead if you specifically want ky only

    klist.append(k)



plt.scatter(klist,np.abs(eigenvalues),s = 1)
plt.xlim(0,0.5e6)
plt.show()