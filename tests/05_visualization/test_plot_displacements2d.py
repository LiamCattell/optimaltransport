import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

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
img1 = np.zeros((128,100))
img1[43,78] = 1.
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
fig2, ax2 = plt.subplots(1, 3, sharex=True, sharey=True)
ax2[0].imshow(clot.displacements_[0])
ax2[1].imshow(clot.displacements_[1])
plot_displacements2d(clot.displacements_, ax2[2], scale=5000., count=30)
fig2.tight_layout()

plt.show()
