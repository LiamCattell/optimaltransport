import sys
sys.path.append('../../../optimaltransport')

import numpy as np
import matplotlib.pyplot as plt

from optrans.utils import signal_to_pdf
from optrans.continuous import CDT

"""
Compute the forward and inverse CDT of a Gaussian w.r.t a uniform distribution.

Liam Cattell -- January 2018
"""

# Create the uniform reference and Gaussian signal
x = np.linspace(-5,5,256)
sig0 = np.ones(x.size)
sig1 = 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2)

# Convert signals to PDFs
sig0 = signal_to_pdf(sig0)
sig1 = signal_to_pdf(sig1)

# Compute CDT of sig1 w.r.t sig0
cdt = CDT()
sig1_hat = cdt.forward(sig0, sig1)

# Apply transport map in order to reconstruct signals
sig0_recon = cdt.apply_forward_map(cdt.transport_map_, sig1)
sig1_recon = cdt.apply_inverse_map(cdt.transport_map_, sig0)
# sig1_recon = cdt.inverse()

# Plot results
fig, ax = plt.subplots(3, 2, sharex=True)
ax[0,0].plot(sig0)
ax[0,0].set_title('Reference sig0')
ax[0,1].plot(sig1)
ax[0,1].set_title('Signal sig1')
ax[1,0].plot(sig0_recon)
ax[1,0].set_title('Reconstructed sig0')
ax[1,1].plot(sig1_recon)
ax[1,1].set_title('Reconstructed sig1')
ax[2,0].plot(cdt.displacements_)
ax[2,0].set_title('Displacements u')
ax[2,1].plot(sig1_hat)
ax[2,1].set_title('CDT')

plt.show()
