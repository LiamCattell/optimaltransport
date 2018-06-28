import sys
sys.path.append('../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT

"""
This is the example in the README.

Liam Cattell -- June 2018
"""

# Create a reference image that integrates to 1
img0 = np.zeros((128,128))
img0[96,24] = 1.
img0 = signal_to_pdf(img0, sigma=5., total=1.)

# Create an image that also integrates to 1
img1 = np.zeros((128,128))
img1[32,78] = 1.
img1 = signal_to_pdf(img1, sigma=5., total=1.)

# Compute the forward transform of img1 w.r.t. img0
radoncdt = RadonCDT()
rcdt = radoncdt.forward(img0, img1)

# Reconstruct the original images
img1_recon = radoncdt.inverse()
img0_recon = radoncdt.apply_forward_map(radoncdt.transport_map_, img1)

# Plot the results
fig, ax = plt.subplots(3, 2, figsize=(6,8))
ax[0,0].imshow(img0)
ax[0,0].set_title('Reference')
ax[0,1].imshow(img1)
ax[0,1].set_title('Image')
ax[1,0].imshow(img0_recon)
ax[1,0].set_title('Reconstructed\nreference')
ax[1,1].imshow(img1_recon)
ax[1,1].set_title('Reconstructed\nimage')
ax[2,0].imshow(radoncdt.displacements_)
ax[2,0].set_title('Displacements')
ax[2,1].imshow(radoncdt.transport_map_)
ax[2,1].set_title('Transport map')
fig.tight_layout()
plt.show()
