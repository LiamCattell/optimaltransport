import sys
sys.path.append('../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from optrans.continuous import CLOT
from optrans.datasets import adni
from optrans.utils import signal_to_pdf, interp2d, griddata2d

n = 128
epsilon = 1e-8
total = 1.
xv, yv = np.meshgrid(np.arange(n), np.arange(n))

# img1 = np.zeros((n,n))
# # img1[n//2,n//2] = 1
# img1[90,30] = 1
# img1[30,90] = 1
# img0 = np.ones_like(img1)

# data = loadmat("C:/Users/lcc3d/Documents/MATLAB/cmu_code/ContinuousLOT/ContinuousLOT/Data/FaceSamples.mat")
data = loadmat("C:/Users/lcc3d/Documents/MATLAB/cmu_code/ContinuousLOT/ContinuousLOT/Data/FaceNormalized.mat")
img0 = data['I'][0][0].astype(np.float)
img1 = data['I'][0][1].astype(np.float)

# img0 = signal_to_pdf(img0, sigma=2.5, epsilon=epsilon, total=total)
# img1 = signal_to_pdf(img1, sigma=2.5, epsilon=epsilon, total=total)

print('SUM ', img0.sum())
print(img0.min(), img0.max())

clot = CLOT(max_iter=300, lr=1e-6, verbose=2)
lot = clot.forward(img0, img1)

img0_recon0 = clot.apply_forward_map(clot.transport_map_initial_, img1)
img1_recon0 = clot.apply_inverse_map(clot.transport_map_initial_, img0)

img0_recon = clot.apply_forward_map(clot.transport_map_, img1)
img1_recon = clot.apply_inverse_map(clot.transport_map_, img0)

vmin = min(img0.min(), img1.min())
vmax = max(img0.max(), img1.max())

fig1, ax1 = plt.subplots(3, 2)
ax1[0,0].imshow(img0, vmin=vmin, vmax=vmax)
ax1[0,1].imshow(img1, vmin=vmin, vmax=vmax)
ax1[1,0].imshow(img0_recon0, vmin=vmin, vmax=vmax)
ax1[1,1].imshow(img1_recon0, vmin=vmin, vmax=vmax)
ax1[2,0].imshow(img0_recon, vmin=vmin, vmax=vmax)
ax1[2,1].imshow(img1_recon, vmin=vmin, vmax=vmax)

fig2, ax2 = plt.subplots(3, 2)
ax2[0,0].imshow(clot.displacements_initial_[0])
ax2[0,1].imshow(clot.displacements_initial_[1])
ax2[1,0].imshow(clot.displacements_[0])
ax2[1,1].imshow(clot.displacements_[1])
ax2[2,0].imshow(clot.displacements_initial_[0]-clot.displacements_[0])
ax2[2,1].imshow(clot.displacements_initial_[1]-clot.displacements_[1])
plt.show()
