import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from optrans.continuous import CLOT
from optrans.utils import signal_to_pdf
from optrans.visualization import plot_displacements2d

"""
Plot 2D pixel displacement map as a wireframe grid.

Liam Cattell -- May 2018
"""

# Image normalization parameters
sigma = 5.
epsilon = 8.
total = 100.

# Create sample images
img1 = np.zeros((128,128))
img1[32,64] = 1.
img0 = np.ones_like(img1)

# CLOT is *very* sensitive to the image normalization
img0 = signal_to_pdf(img0, sigma=sigma, epsilon=epsilon, total=total)
img1 = signal_to_pdf(img1, sigma=sigma, epsilon=epsilon, total=total)

# Continuous LOT
clot = CLOT(max_iter=500, lr=1e-5, momentum=0.9, verbose=1)
# clot = VOT2D(lr=0.0001, verbose=1)
lot = clot.forward(img0, img1)

# Reconstruct images using final map
img0_recon = clot.apply_forward_map(clot.transport_map_, img1)
img1_recon = clot.apply_inverse_map(clot.transport_map_, img0)

# Colour scaling
vmin = min(img0.min(), img1.min())
vmax = max(img0.max(), img1.max())

# Plot images and reconstructions
fig1, ax1 = plt.subplots(2, 2)
ax1[0,0].imshow(img0, vmin=vmin, vmax=vmax)
ax1[0,0].set_title("Ref.")
ax1[0,1].imshow(img1, vmin=vmin, vmax=vmax)
ax1[0,1].set_title("Image")
ax1[1,0].imshow(img0_recon, vmin=vmin, vmax=vmax)
ax1[1,0].set_title("Ref. recon.")
ax1[1,1].imshow(img1_recon, vmin=vmin, vmax=vmax)
ax1[1,1].set_title("Image recon.")
fig1.tight_layout()

# Plot initial and final transport maps
fig2 = plt.figure()
ax2a = fig2.add_subplot(1, 3, 1)
ax2a.imshow(clot.displacements_[0])
ax2a.set_title("v")
ax2b = fig2.add_subplot(1, 3, 2)
ax2b.imshow(clot.displacements_[1])
ax2b.set_title("u")
ax2c = fig2.add_subplot(1, 3, 3, projection='3d')
ax2c = plot_displacements2d(clot.displacements_, ax=ax2c, scale=5000, count=30)
ax2c.dist = 6
ax2c.set_title("Displacements")
fig2.tight_layout()
plt.show()
