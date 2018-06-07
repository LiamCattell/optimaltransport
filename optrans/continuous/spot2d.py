import numpy as np
from skimage.transform import pyramid_reduce, pyramid_expand, resize
from scipy.ndimage import gaussian_filter

from .base import BaseTransform, BaseMapper2D
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, interp2d, griddata2d


class SPOT2D(BaseMapper2D):
    """
    Single-Potential Optimal Transport 2D Transform.

    Parameters
    ----------
    alpha : float (default=0.01)
        Regularization of the curl operator.
    lr : float (default=0.01)
        Learning rate.
    beta1 : float (default=0.9)
        Adam optimizer parameter. 0 < beta1 < 1. Generally close to 1.
    beta2 : float (default=0.999)
        Adam optimizer parameter. 0 < beta2 < 1. Generally close to 1.
    decay : float (default=0.)
        Learning rate decay over each update.
    max_iter : int (default=300)
        Maximum number of iterations.
    tol : float (default=0.001)
        Stop iterating when change in cost function is below this threshold.
    verbose : int (default=1)
        Verbosity during optimization. 0=no output, 1=print cost,
        2=print all metrics.

    Attributes
    -----------
    displacements_ : array, shape (2, height, width)
        Displacements u. First index denotes direction: displacements_[0] is
        y-displacements, and displacements_[1] is x-displacements.
    transport_map_ : array, shape (2, height, width)
        Transport map f. First index denotes direction: transport_map_[0] is
        y-map, and transport_map_[1] is x-map.
    cost_ : list of float
        Value of cost function at each iteration.
    mse_ : list of float
        Mean squared error at each iteration.
    curl_ : list of float
        Curl at each iteration.

    References
    ----------
    [Transport-based pattern theory: A signal transformation approach]
    (https://arxiv.org/abs/1802.07163)
    [Adam - A method for stochastic optimization]
    (http://arxiv.org/abs/1412.6980v8)
    """
    def __init__(self, sigma=2., lr=0.01, beta1=0.9, beta2=0.999, decay=0.,
                 max_iter=300, tol=0.001, verbose=0):
        super(SPOT2D, self).__init__()
        self.sigma = sigma
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        return


    def forward(self, sig0, sig1):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        lot : array, shape (2, height, width)
            LOT transform of input image sig1. First index denotes direction:
            lot[0] is y-LOT, and lot[1] is x-LOT.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Set reference signal
        self.sig0_ = sig0

        # Initialise regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # Set the fill value for interpolation
        fill_val = min(sig0.min(), sig1.min())

        # Initialise evaluation measures
        self.cost_ = []
        self.mse_ = []
        self.curl_ = []

        # Initialise coefficients and transport map
        c = np.zeros_like(sig0)
        f = np.zeros((2,h,w), dtype=float)

        # Initialise Adam moment estimates
        mt = np.zeros_like(c)
        vt = np.zeros_like(c)

        # Iterate!
        for i in range(self.max_iter):
            f[0] = yv - gaussian_filter(c, self.sigma, order=[1,0])
            f[1] = xv - gaussian_filter(c, self.sigma, order=[0,1])

            # Jacobian and its determinant
            f0y = 1. - gaussian_filter(c, self.sigma, order=[2,0])
            f0x = -gaussian_filter(c, self.sigma, order=1)
            f1y = -gaussian_filter(c, self.sigma, order=1)
            f1x = 1. - gaussian_filter(c, self.sigma, order=[0,2])
            detJ = (f1x * f0y) - (f1y * f0x)

            # Transform sig1 using f (i.e. sig1(f))
            sig1f = interp2d(sig1, f, fill_value=fill_val)
            sig1fy, sig1fx = np.gradient(sig1f)

            # Reconstructed sig0 and its error
            sig0_recon = detJ * sig1f
            err = sig0_recon - sig0

            # Update evaluation metrics
            curl = 0.5 * (f0x - f1y)
            self.cost_.append(np.sum(err**2))
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

            # Compute derivative of cost function w.r.t. c
            tmp0 = np.rot90(err*detJ*sig1fy, 2)
            tmp1 = np.rot90(err*detJ*sig1fx, 2)
            g0 = np.rot90(gaussian_filter(tmp0, self.sigma, order=[1,0]), 2)
            g1 = np.rot90(gaussian_filter(tmp1, self.sigma, order=[0,1]), 2)
            g2 = gaussian_filter(err*f1y*sig1f, self.sigma, order=1)
            g3 = gaussian_filter(err*f0x*sig1f, self.sigma, order=1)
            g4 = gaussian_filter(err*f1x*sig1f, self.sigma, order=[2,0])
            g5 = gaussian_filter(err*f0y*sig1f, self.sigma, order=[0,2])
            ct = -g0 - g1 + g2 + g3 - g4 - g5

            # Save previous version of c before update
            c_prev = np.copy(c)

            # Update c using Adam optimizer
            self.lr *= 1. / (1. + self.decay*i)
            lrt = self.lr * np.sqrt(1-self.beta2**i) / (1-self.beta1)
            mt = self.beta1 * mt + (1-self.beta1) * ct
            vt = self.beta2 * vt + (1-self.beta2) * ct**2
            update = lrt * mt / (np.sqrt(vt) + 1e-8)
            c -= update

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
        self.potential_ = c_prev
        self.transport_map_ = f
        self.displacements_ = f - np.stack((yv,xv))
        lot = self.displacements_ * np.sqrt(sig0)

        self.is_fitted = True

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
        return self.apply_inverse_potential(self.transport_map_, self.sig0_)


    def apply_forward_potential(self, potential, sig1, sigma):
        """
        Appy forward transport map derived from potential.

        Parameters
        ----------
        potential : array, shape (height, width)
            Potential.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        sig0_recon : array, shape (height, width)
            Reconstructed reference signal sig0.
        """
        # Check inputs
        potential = check_array(potential, ndim=2,
                                dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(potential, sig1, ['potential', 'sig1'])

        # Regular grid
        h, w = sig1.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # Transport map
        f = np.zeros((2,h,w), dtype=float)
        f[0] = yv - gaussian_filter(potential, sigma, order=[1,0])
        f[1] = xv - gaussian_filter(potential, self.sigma, order=[0,1])

        # Jacobian and its determinant
        f0y = 1. - gaussian_filter(potential, sigma, order=[2,0])
        f0x = -gaussian_filter(potential, sigma, order=1)
        f1y = -gaussian_filter(potential, sigma, order=1)
        f1x = 1. - gaussian_filter(potential, sigma, order=[0,2])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Reconstruct sig0
        sig0_recon = detJ * interp2d(sig1, f, fill_value=sig1.min())

        return sig0_recon


    def apply_inverse_potential(self, potential, sig0, sigma):
        """
        Appy inverse transport map derived from potential.

        Parameters
        ----------
        potential : array, shape (height, width)
            Potential. Inverse transport map is computed in this function.
        sig0 : array, shape (height, width)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        # Check inputs
        potential = check_array(potential, ndim=2,
                                dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(potential, sig0, ['potential', 'sig0'])

        # Regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # Transport map
        f = np.zeros((2,h,w), dtype=float)
        f[0] = yv - gaussian_filter(potential, sigma, order=[1,0])
        f[1] = xv - gaussian_filter(potential, self.sigma, order=[0,1])

        # Jacobian and its determinant
        f0y = 1. - gaussian_filter(potential, sigma, order=[2,0])
        f0x = -gaussian_filter(potential, sigma, order=1)
        f1y = -gaussian_filter(potential, sigma, order=1)
        f1x = 1. - gaussian_filter(potential, sigma, order=[0,2])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Let's hope there are no NaNs/Infs in sig0/detJ
        sig1_recon = griddata2d(sig0/detJ, f, fill_value=sig0.min())

        return sig1_recon
