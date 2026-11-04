"""worldsim: физическая «реальность» для обучения и квалификации world models."""
from .client import Client, RemoteEnv
from .data import Dataset
from .mission import MissionRunner
from .model import FunctionModel, Persistence, WorldModel
from .qualify import qualify, to_markdown
from .vec import VectorEnv



def fit_telemetry(*args, **kwargs):
    """Калибровка по записи полёта, см. worldsim.calibrate. Импорт отложен, чтобы `python -m worldsim.calibrate` работал чисто."""
    from .calibrate import calibrate
    return calibrate(*args, **kwargs)


__all__ = ["MissionRunner", "fit_telemetry", "Client", "RemoteEnv", "Dataset", "FunctionModel", "Persistence", "WorldModel", "VectorEnv", "qualify", "to_markdown"]
