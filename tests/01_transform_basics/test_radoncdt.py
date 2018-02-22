import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

from optrans.utils import signal_to_pdf
from optrans.continuous import RadonCDT

"""
Compute the forward and inverse Radon-CDT of a sample image.

Liam Cattell -- January 2018
"""

# Load a sample image
img = img_as_float(data.camera()[::2,::2])

# Select two patches to be our sample images
img0 = img[50:162,70:134]
img1 = img[32:144,64:128]

# Convert images to PDFs
img0 = signal_to_pdf(img0, sigma=1.)
img1 = signal_to_pdf(img1, sigma=1.)

# Compute Radon-CDT of img1 w.r.t img0
theta = np.arange(0,179,2)
radoncdt = RadonCDT(theta=theta)
img1_hat = radoncdt.forward(img0, img1)

# Apply transport map in order to reconstruct images
img0_recon = radoncdt.apply_forward_map(radoncdt.transport_map_, img1)
img1_recon = radoncdt.apply_inverse_map(radoncdt.transport_map_, img0)
# img1_recon = radoncdt.inverse()


fig, ax = plt.subplots(3,2)
ax[0,0].imshow(img0)
ax[0,0].set_title('Reference img0')
ax[0,1].imshow(img1)
ax[0,1].set_title('Image img1')
ax[1,0].imshow(img0_recon)
ax[1,0].set_title('Reconstructed img0')
ax[1,1].imshow(img1_recon)
ax[1,1].set_title('Reconstructed img1')
ax[2,0].imshow(radoncdt.displacements_)
ax[2,0].set_title('Displacements u')
ax[2,1].imshow(img1_hat)
ax[2,1].set_title('Radon-CDT')

plt.show()
