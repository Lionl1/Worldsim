"""worldsim: physical “reality” for training and qualification world models."""
from .client import Client, RemoteEnv
from .data import Dataset
from .mission import MissionRunner
from .model import FunctionModel, Persistence, WorldModel
from .planning import planning_check, planning_markdown
from .qualify import qualify, to_markdown
from .vec import VectorEnv



def fit_telemetry(*args, **kwargs):
    """Calibration from a flight recording, see worldsim.calibrate. The import is deferred so that `python -m worldsim.calibrate` runs cleanly."""
    from .calibrate import calibrate
    return calibrate(*args, **kwargs)


__all__ = ["planning_check", "planning_markdown", "MissionRunner", "fit_telemetry", "Client", "RemoteEnv", "Dataset", "FunctionModel", "Persistence", "WorldModel", "VectorEnv", "qualify", "to_markdown"]
