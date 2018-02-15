import numpy as np


class BaseTransform(object):
    """
    Base class for optimal transport transform methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def __init__(self):
        self.is_fitted = False
        self.sig0_ = None
        self.displacements_ = None
        self.transport_map_ = None


    def _check_is_fitted(self):
        if not self.is_fitted:
            raise AssertionError("The forward transform of {0!s} has not been "
                                 "called yet. Call 'forward' before using "
                                 "this method".format(type(self).__name__))


    def forward(self):
        """
        Placeholder for forward transform.
        Subclasses should implement this method!
        """
        raise NotImplementedError


    def inverse(self):
        """
        Placeholder for inverse transform.
        Subclasses should implement this method!
        """
        raise NotImplementedError


    def apply_forward_map(self):
        """
        Placeholder for application of forward transport map.
        Subclasses should implement this method!
        """
        raise NotImplementedError


    def apply_inverse_map(self):
        """
        Placeholder for application of inverse transport map.
        Subclasses should implement this method!
        """
        raise NotImplementedError
