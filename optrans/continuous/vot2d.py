import numpy as np
from scipy.interpolate import interp2d

from .base import BaseTransform
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf


class VOT2D(BaseTransform):
    def __init__(self, alpha=0.01, lr=0.01, momentum=0., decay=0.,
                 max_iter=300, tol=0.001, verbose=0):
        super(VOT2D, self).__init__()
        self.alpha = alpha
        self.lr = lr
        self.momentum = momentum
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

        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        fill_val = min(sig0.min(), sig1.min())

        mask = np.zeros_like(sig0)
        mask[2:-2,2:-2] = 1.

        # f = np.zeros((2,h,w))
        f = np.stack((yv,xv), axis=0)
        objective = []

        interp = interp2d(range(w), range(h), sig1, bounds_error=False, fill_value=fill_val)

        for i in range(self.max_iter):
            print("Iteration ", i+1)
            # Decay learning rate
            self.lr *= 1. / (1. + i*self.decay)

            # Gradient
            f0x, f0y = np.gradient(f[0])
            f1x, f1y = np.gradient(f[1])

            # 2nd derivative
            f0xx, f0xy = np.gradient(f0x)
            f0yx, f0yy = np.gradient(f0y)
            f1xx, f1xy = np.gradient(f1x)
            f1yx, f1yy = np.gradient(f1y)

            detJ = (f1x * f0y) - (f1y * f0x)
            print(f[0])
            print(f[0].flatten().shape)
            sig1f = interp(f[1].ravel(), f[0].ravel())

            err = detJ * sig1f - sig0

            objective.append(0.5*np.sum(err**2)+ self.alpha*np.sum((f0x-f1y)**2))


            # "Look ahead" for Nesterov momentum
