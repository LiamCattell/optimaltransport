import numpy as np
# from scipy.interpolate import griddata

from .base import BaseTransform
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, interp2d, griddata2d


class VOT2D(BaseTransform):
    def __init__(self, alpha=0.01, lr=0.01, beta1=0.9, beta2=0.999, decay=0.,
                 max_iter=300, tol=0.001, verbose=0):
        super(VOT2D, self).__init__()
        self.alpha = alpha
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose


    def forward(self, sig0, sig1):
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Set reference signal
        self.sig0_ = sig0

        # Initialise transport map to be identity transform (i.e. f = x)
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))
        f = np.stack((yv,xv), axis=0)

        # Set the fill value for interpolation
        fill_val = min(sig0.min(), sig1.min())

        # Create a mask to avoid edge effects
        mask = np.zeros_like(sig0)
        mask[2:-2,2:-2] = 1.

        # Initialise evaluation measures
        self.cost_ = []
        self.mse_ = []
        self.curl_ = []

        # Initialise derivative of cost function wrt f
        ft = np.zeros_like(f)

        # Initialise Adam moment estimates
        mt = np.zeros_like(f)
        vt = np.zeros_like(f)

        # Iterate!
        for i in range(self.max_iter):
            # Jacobian and its determinant
            f0y, f0x = np.gradient(f[0])
            f1y, f1x = np.gradient(f[1])
            detJ = (f1x * f0y) - (f1y * f0x)

            # Transform sig1 using f (i.e. sig1(f))
            sig1f = interp2d(sig1, f, fill_value=fill_val)
            sig1fy, sig1fx = np.gradient(sig1f)

            # Reconstructed sig0 and its error
            sig0_recon = detJ * sig1f
            err = sig0_recon - sig0

            # Update evaluation metrics
            cost = 0.5*np.sum(err**2) + self.alpha*np.sum((f0x-f1y)**2)
            curl = 0.5 * (f0x - f1y)
            self.cost_.append(cost)
            self.mse_.append(np.mean(err**2))
            self.curl_.append(0.5*np.sum(curl**2))

            # Print cost value
            if self.verbose:
                print('Iteration {:>4} -- '
                      'cost = {:.4e}'.format(i, self.cost_[-1]))

            # Print MSE and curl
            if self.verbose > 1:
                print('... mse = {:.4e}, '
                      'curl = {:.4e}'.format(self.mse_[-1], self.curl_[-1]))

            # Useful 2nd derivatives
            f0xy, f0xx = np.gradient(f0x)
            f1yy, f1yx = np.gradient(f1y)

            # Compute divergence
            _, g0x = np.gradient(-f1y*err*sig1f)
            g0y, _ = np.gradient(f1x*err*sig1f)
            _, g1x = np.gradient(f0y*err*sig1f)
            g1y, _ = np.gradient(-f0x*err*sig1f)
            div0 = g0x + g0y
            div1 = g1x + g1y

            # Derivative of cost function wrt f
            ft[0] = detJ * sig1fy * err - div0 + self.alpha*(f1yx-f0xx)
            ft[1] = detJ * sig1fx * err - div1 + self.alpha*(f0xy-f1yy)

            # Mask the derivative to avoid edge effects
            ft *= mask

            # Save previous version of f before update
            f_prev = np.copy(f)

            # Update f using Adam optimizer
            self.lr *= 1. / (1. + self.decay*i)
            lrt = self.lr * np.sqrt(1-self.beta2**i) / (1-self.beta1)
            mt = self.beta1 * mt + (1-self.beta1) * ft
            vt = self.beta2 * vt + (1-self.beta2) * ft**2
            update = lrt * mt / (np.sqrt(vt) + 1e-8)
            f -= update

            # If change in cost is below threshold, stop iterating
            if i > 7 and \
                (self.cost_[i-7]-self.cost_[i])/self.cost_[0] < self.tol:
                break

        # Print final evaluation metrics
        if self.verbose:
            print('FINAL METRICS:')
            print('-- cost = {:.4e}'.format(self.cost_[-1]))
            print('-- mse  = {:.4e}'.format(self.mse_[-1]))
            print('-- curl = {:.4e}'.format(self.curl_[-1]))

        # Set final transport map, displacements, and LOT transform
        # Note: Use previous version of f, just in case something weird
        # happened in the final iteration
        self.transport_map_ = f_prev
        self.displacements_ = f_prev - np.stack((yv,xv))
        lot = self.displacements_ * np.sqrt(sig0)

        return lot


    def inverse(self):
        """
        Inverse transform.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        self._check_is_fitted()
        return self.apply_inverse_map(self.transport_map_, self.sig0_)


    def apply_forward_map(self, transport_map, sig1):
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig1, ['transport_map', 'sig1'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Reconstruct sig0
        sig0_recon = detJ * interp2d(sig1, transport_map, fill_value=sig1.min())

        return sig0_recon


    def apply_inverse_map(self, transport_map, sig0):
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig0, ['transport_map', 'sig0'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        sig1_recon = griddata2d(sig0/detJ, transport_map, fill_value=sig0.min())

        return sig1_recon
