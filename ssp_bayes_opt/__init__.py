import importlib.util
import sys
import warnings

if importlib.util.find_spec("nengo_spinnaker") is not None and sys.version_info[:2] != (3, 8):
    warnings.warn(
        "The ssp-bayes-opt[spinnaker] extra installs nengo-spinnaker, which was only "
        "tested on Python 3.8. You are running Python "
        f"{sys.version_info.major}.{sys.version_info.minor}, so there may be problems.",
        UserWarning,
        stacklevel=2,
    )

from .bayesian_optimization import BayesianOptimization
from .nengo_bayesian_optimization import NengoBayesianOptimization
from .network_solver import make_network
from .agents import *