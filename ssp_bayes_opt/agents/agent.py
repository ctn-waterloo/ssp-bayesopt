import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from .. import sspspace
from .. import blr

import functools
import warnings

from scipy.stats import qmc


class Agent:
    """Abstract base class for all BO agents.

    Subclasses must override eval(), update(), and acquisition_func().
    """

    def __init__(self):
        pass

    def eval(self, xs):
        """Evaluate the surrogate at xs, returning (mu, var, acquisition_value)."""
        pass

    def update(self, x_t, y_t, sigma_t, step_num=0):
        """Incorporate a new observation (x_t, y_t) into the surrogate."""
        pass

    def untrusted(self, x, badness=None):
        """Mark region x as untrusted (penalise in acquisition function)."""
        pass

    def acquisition_func(self):
        """Return (objective_fn, jacobian_fn) suitable for scipy.optimize.minimize."""
        pass

    def length_scale(self):
        """Return the current length scale(s) used by the surrogate."""
        pass
