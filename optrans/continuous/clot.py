import numpy as np

from .base import BaseTransform
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, interp2d, griddata2d
