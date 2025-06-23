import numpy as np

r1 = np.array([-0.5,0,0])
r2 = np.array([0.5,0,0])

r12 = 10e-7  ## 10nm in cm

n12 = r2-r1
n21 = r1-r2

p1x_hat = np.array([1,0,0])
p1y_hat = np.array([0,1,0])
p1z_hat = np.array([0,0,1])

p2x_hat = np.array([1,0,0])
p2y_hat = np.array([0,1,0])
p2z_hat = np.array([0,0,1])


