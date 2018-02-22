import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.continuous import CDT
from optrans.utils import signal_to_pdf

"""
Compute the geodesic between two 1D Gaussians.

Liam Cattell -- January 2018
"""

# Create two translated delta functions as test signals
n = 128
sig0 = np.zeros(n)
sig1 = np.zeros(n)
sig0[32] = 1.
sig1[96] = 1.

# Smooth the delta functions into Gaussians and convert to PDFs
sig0 = signal_to_pdf(sig0, sigma=6.)
sig1 = signal_to_pdf(sig1, sigma=6.)

# Compute the CDT of sig1 w.r.t. sig0
cdt = CDT()
sig1_hat = cdt.forward(sig0, sig1)

# Get the domain of our signal
x = cdt.transport_map_ + cdt.displacements_

# Plot linear interpolation in signal space and CDT space
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10,6))
for i,alpha in enumerate(np.linspace(0, 1, 5)):
    # Interpolation in signal space
    sig_interp = (1. - alpha) * sig0 + alpha * sig1
    ax[0,i].plot(sig_interp)
    ax[0,i].set_title('alpha = {:.2f}'.format(alpha))

    # Interpolation in CDT space
    u = cdt.displacements_ * alpha
    f = x - u
    sig_recon = cdt.apply_inverse_map(f, sig0)
    ax[1,i].plot(sig_recon)

ax[0,0].set_ylabel('Signal space')
ax[1,0].set_ylabel('CDT space')
plt.show()
