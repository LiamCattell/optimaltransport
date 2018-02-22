import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import RadonCDT
from optrans.utils import signal_to_pdf

"""
Compute the geodesic between two 2D Gaussians.

Liam Cattell -- January 2018
"""

# Create two translated delta functions as test signals
n = 128
img0 = np.zeros((n,n))
img1 = np.zeros((n,n))
img0[96,32] = 1.
img1[32,96] = 1.

# Smooth the delta functions into Gaussians and convert to PDFs
img0 = signal_to_pdf(img0, sigma=6.)
img1 = signal_to_pdf(img1, sigma=6.)

# Compute the Radon-CDT of img1 w.r.t. img0
radoncdt = RadonCDT()
rcdt = radoncdt.forward(img0, img1)

# Get the domain of our signal
x = radoncdt.transport_map_ + radoncdt.displacements_

# Get min/max intensities for plotting purposes
vmin = img0.min()
vmax = img0.max()

# Plot linear interpolation in signal space and Radon-CDT space
fig, ax = plt.subplots(2, 5, figsize=(10,6))
for i,alpha in enumerate(np.linspace(0, 1, 5)):
    # Interpolation in image space
    img_interp = (1. - alpha) * img0 + alpha * img1
    ax[0,i].imshow(img_interp, vmin=vmin, vmax=vmax)
    ax[0,i].set_title('alpha = {:.2f}'.format(alpha))

    # Interpolation in Radon-CDT space
    u = radoncdt.displacements_ * alpha
    f = x - u
    img_recon = radoncdt.apply_inverse_map(f, img0)
    ax[1,i].imshow(img_recon, vmin=vmin, vmax=vmax)

ax[0,0].set_ylabel('Image space')
ax[1,0].set_ylabel('Radon-CDT space')
plt.show()
