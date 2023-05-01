from typing import List, Tuple
from abc import ABCMeta, abstractmethod


class EnvArguments(metaclass=ABCMeta):
    """Arguments struct"""
    pass


class Env(metaclass=ABCMeta):
    """"""
    args: EnvArguments = None

    def __init__(self, args: EnvArguments):
        self.args = args

    @abstractmethod
    def sample(self, x: List[float]):
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> List[float]:
        raise NotImplementedError()

    @abstractmethod
    def step(self, a: List[float]) -> Tuple[List[float], float, bool, dict]:
        raise NotImplementedError()
