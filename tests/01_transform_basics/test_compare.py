import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.datasets import oasis
from optrans.utils import signal_to_pdf
from optrans.visualization import plot_displacements2d
from optrans.continuous import RadonCDT, VOT2D, MultiVOT2D, CLOT, SPOT2D

"""
A comparison of different 2D continuous optimal transport methods:
- RadonCDT
- VOT2D
- MultiVOT2D
- CLOT
- SPOT2D

Liam Cattell -- June 2018
"""

# Load sample data
img0 = oasis.load_img0()
X, _, _ = oasis.load_data()
img1 = X[0]

# Convert images to PDFs
img0 = signal_to_pdf(img0, sigma=1., total=100.)
img1 = signal_to_pdf(img1, sigma=1., total=100.)

# Initialise optimal transport methods
methods = [RadonCDT(),
           VOT2D(alpha=0.001),
           MultiVOT2D(alpha=0.001),
           CLOT(lr=1e-7, tol=1e-4),
           SPOT2D()]

# Plot settings
fs = 16
vmin = min(img0.min(), img1.min())
vmax = max(img0.max(), img1.max())

# Initialise figure and plot original images
fig, ax = plt.subplots(3, len(methods)+1, sharex=True, sharey=True,
                       figsize=(12,6))
ax[0,0].imshow(img0, vmin=vmin, vmax=vmax)
ax[0,0].set_frame_on(False)
ax[0,0].set_ylabel("$I_0$", fontsize=fs)
ax[0,0].set_title("Original", fontsize=fs)
ax[1,0].imshow(img1, vmin=vmin, vmax=vmax)
ax[1,0].set_frame_on(False)
ax[1,0].set_ylabel("$I_1$", fontsize=fs)
ax[2,0].axis('off')

# Loop over each method
for i,method in enumerate(methods, 1):
    name = method.__class__.__name__

    # Compute LOT transform and reconstruct images using transport map
    lot = method.forward(img0, img1)
    img0_recon = method.apply_forward_map(method.transport_map_, img1)
    img1_recon = method.apply_inverse_map(method.transport_map_, img0)

    # Plot images and deformation field (where possible)
    if name == 'RadonCDT':
        ax[0,i].imshow(img0_recon)
        ax[1,i].imshow(img1_recon)
    else:
        ax[0,i].imshow(img0_recon, vmin=vmin, vmax=vmax)
        ax[1,i].imshow(img1_recon, vmin=vmin, vmax=vmax)
        plot_displacements2d(method.displacements_, ax=ax[2,i], count=20)

    # Set title and labels
    ax[0,i].set_title(name, fontsize=fs)
    if i == 1:
        ax[0,i].set_ylabel("$I_0 = D_{f} I_1 \circ f$", fontsize=fs)
        ax[1,i].set_ylabel("$I_1 = D_{f^{-1}} I_0 \circ f^{-1}$", fontsize=fs)

    # Remove frame on plots
    ax[0,i].set_frame_on(False)
    ax[1,i].set_frame_on(False)
    ax[2,i].set_frame_on(False)

# Display figure
fig.tight_layout()
plt.show()
