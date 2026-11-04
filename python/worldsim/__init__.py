"""worldsim: физическая «реальность» для обучения и квалификации world models."""
from .client import Client, RemoteEnv
from .data import Dataset
from .model import FunctionModel, Persistence, WorldModel
from .qualify import qualify, to_markdown
from .vec import VectorEnv

__all__ = ["Client", "RemoteEnv", "Dataset", "FunctionModel", "Persistence", "WorldModel", "VectorEnv", "qualify", "to_markdown"]
