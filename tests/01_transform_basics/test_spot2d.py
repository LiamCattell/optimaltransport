import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import SPOT2D
from optrans.datasets import adni

"""
Compute the transport map between two images using the single-scale single
potential optimal transport method.

Liam Cattell -- May 2018
"""

# Load some images
X, _ = adni.load_data()
img0 = adni.load_img0()
img1 = X[1]

# Variational OT
sigma = 2.
spot = SPOT2D(lr=0.01, sigma=sigma, max_iter=1000, verbose=2)
img1_hat = spot.forward(img0, img1)

# Reconstruct images using transport map
img0_recon1 = spot.apply_forward_map(spot.transport_map_, img1)
img1_recon1 = spot.apply_inverse_map(spot.transport_map_, img0)
img0_recon2 = spot.apply_forward_potential(spot.potential_, img1, sigma=sigma)
img1_recon2 = spot.apply_inverse_potential(spot.potential_, img0, sigma=sigma)

# Plot cost function and evaluation metrics
fig1, ax1 = plt.subplots(1, 3, sharex=True)
ax1[0].plot(spot.cost_)
ax1[0].set_title('Cost')
ax1[1].plot(spot.mse_)
ax1[1].set_title('MSE')
ax1[2].plot(spot.curl_)
ax1[2].set_title('Curl')
fig1.tight_layout()

# Plot images and reconstructions
fig2, ax2 = plt.subplots(3, 2)
ax2[0,0].imshow(img0)
ax2[0,0].set_title("Ref.")
ax2[0,1].imshow(img1)
ax2[0,1].set_title("Image")
ax2[1,0].imshow(img0_recon1)
ax2[1,0].set_title("Ref. recon. (map)")
ax2[1,1].imshow(img1_recon1)
ax2[1,1].set_title("Image recon. (map)")
ax2[2,0].imshow(img0_recon2)
ax2[2,0].set_title("Ref. recon. (pot.)")
ax2[2,1].imshow(img1_recon2)
ax2[2,1].set_title("Image recon. (pot.)")
fig2.tight_layout()

# Plot transport map displacements
fig3, ax3 = plt.subplots(1, 2)
ax3[0].imshow(spot.displacements_[0])
ax3[0].set_title("v")
ax3[1].imshow(spot.displacements_[1])
ax3[1].set_title("u")
fig3.tight_layout()

plt.show()
