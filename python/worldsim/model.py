"""Что должна уметь модель, чтобы её можно было квалифицировать.

Подходит любая архитектура: состояние модели непрозрачно для оценщика (вектор наблюдения, RSSM, латент JEPA…).
Единственное требование — уметь выдавать предсказание в каналах сенсоров; для чисто латентных моделей
это обычно обученная поверх латента линейная «проба».
"""
import numpy as np


class WorldModel:
    """Пакетный интерфейс: все массивы имеют первую размерность N (число параллельных траекторий)."""

    name = "model"

    def init(self, obs, params):
        """Начальное состояние модели по наблюдению (N×O) и известным параметрам мира (N×P)."""
        return np.asarray(obs, dtype=np.float32).copy()

    def step(self, state, action):
        """Один шаг в воображении: вернуть (новое состояние, предсказанное наблюдение N×O)."""
        raise NotImplementedError


class FunctionModel(WorldModel):
    """Модель без памяти: next_obs = fn(obs, action, params)."""

    def __init__(self, fn, name="function"):
        self.fn, self.name, self._params = fn, name, None

    def init(self, obs, params):
        self._params = params
        return np.asarray(obs, dtype=np.float32).copy()

    def step(self, state, action):
        n = np.asarray(self.fn(state, action, self._params), dtype=np.float32)
        return n, n


class Persistence(WorldModel):
    """Нулевая модель «ничего не меняется». Любая настоящая модель должна быть лучше."""

    name = "persistence"

    def step(self, state, action):
        return state, state
