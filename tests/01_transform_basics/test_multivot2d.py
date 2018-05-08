import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import MultiVOT2D
from optrans.datasets import adni

"""
Compute the transport map between two images using the multi-scale variational
optimal transport method.

Liam Cattell -- May 2018
"""

# Load images
X, _ = adni.load_data()
img0 = adni.load_img0()
img1 = X[1]

# Multi-scale Variational OT
n_scales = 3
vot = MultiVOT2D(n_scales=n_scales, lr=0.01, alpha=0., max_iter=300, verbose=2)
img1_hat = vot.forward(img0, img1)

# Reconstruct images using transport map
img0_recon = vot.apply_forward_map(vot.transport_map_, img1)
img1_recon = vot.apply_inverse_map(vot.transport_map_, img0)

# Plot cost function for each scale
fig1, ax1 = plt.subplots(n_scales, 3)
for i,a in enumerate(ax1.reshape((-1,3))):
    a[0].plot(vot.cost_all_[i])
    a[0].set_title('Cost {}'.format(i))
    a[1].plot(vot.mse_all_[i])
    a[1].set_title('MSE {}'.format(i))
    a[2].plot(vot.curl_all_[i])
    a[2].set_title('Curl {}'.format(i))
fig1.tight_layout()

# Plot images and reconstructions
fig2, ax2 = plt.subplots(2, 2)
ax2[0,0].imshow(img0)
ax2[0,0].set_title('Ref.')
ax2[0,1].imshow(img1)
ax2[0,1].set_title('Image')
ax2[1,0].imshow(img0_recon)
ax2[1,0].set_title('Ref. recon.')
ax2[1,1].imshow(img1_recon)
ax2[1,1].set_title('Image recon.')
fig2.tight_layout()

# Plot displacements for each scale
fig3, ax3 = plt.subplots(n_scales, 2)
for i,a in enumerate(ax3.reshape((-1,2))):
    a[0].imshow(vot.displacements_all_[i][0])
    a[0].set_title('u {}'.format(i))
    a[1].imshow(vot.displacements_all_[i][1])
    a[1].set_title('v {}'.format(i))
fig3.tight_layout()

plt.show()
