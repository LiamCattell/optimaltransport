import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import CLOT
from optrans.datasets import adni
from optrans.utils import signal_to_pdf

"""
Compute the transport map between two images using the continuous linear optimal
transport method.

Liam Cattell -- May 2018
"""

# Normalization parameters
sigma = 1.
epsilon = 8.
total = 100.

# Load images
X, _ = adni.load_data()
img0 = adni.load_img0()
img1 = X[1]

# CLOT is *very* sensitive to the image normalization
img0 = signal_to_pdf(img0, sigma=sigma, epsilon=epsilon, total=total)
img1 = signal_to_pdf(img1, sigma=sigma, epsilon=epsilon, total=total)

# Continuous LOT
clot = CLOT(max_iter=500, lr=1e-5, momentum=0.9, verbose=1)
lot = clot.forward(img0, img1)

# Reconstruct images using initial map
img0_recon0 = clot.apply_forward_map(clot.transport_map_initial_, img1)
img1_recon0 = clot.apply_inverse_map(clot.transport_map_initial_, img0)

# Reconstruct images using final map
img0_recon = clot.apply_forward_map(clot.transport_map_, img1)
img1_recon = clot.apply_inverse_map(clot.transport_map_, img0)

# Colour scaling
vmin = min(img0.min(), img1.min())
vmax = max(img0.max(), img1.max())

# Plot images and reconstructions
fig1, ax1 = plt.subplots(3, 2)
ax1[0,0].imshow(img0, vmin=vmin, vmax=vmax)
ax1[0,0].set_title("Ref.")
ax1[0,1].imshow(img1, vmin=vmin, vmax=vmax)
ax1[0,1].set_title("Image")
ax1[1,0].imshow(img0_recon0, vmin=vmin, vmax=vmax)
ax1[1,0].set_title("Ref. recon. 0")
ax1[1,1].imshow(img1_recon0, vmin=vmin, vmax=vmax)
ax1[1,1].set_title("Image recon. 0")
ax1[2,0].imshow(img0_recon, vmin=vmin, vmax=vmax)
ax1[2,0].set_title("Ref. recon.")
ax1[2,1].imshow(img1_recon, vmin=vmin, vmax=vmax)
ax1[2,1].set_title("Image recon.")
fig1.tight_layout()

# Plot initial and final transport maps
fig2, ax2 = plt.subplots(2, 2)
ax2[0,0].imshow(clot.displacements_initial_[0])
ax2[0,0].set_title("v0")
ax2[0,1].imshow(clot.displacements_initial_[1])
ax2[0,1].set_title("u0")
ax2[1,0].imshow(clot.displacements_[0])
ax2[1,0].set_title("v")
ax2[1,1].imshow(clot.displacements_[1])
ax2[1,1].set_title("u")
fig2.tight_layout()

fig3, ax3 = plt.subplots(1, 2)
ax3[0].plot(clot.cost_)
ax3[0].set_title("Cost")
ax3[1].plot(clot.curl_)
ax3[1].set_title("Curl")
fig3.tight_layout()

plt.show()
