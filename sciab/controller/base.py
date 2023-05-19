from typing import List
from abc import ABCMeta, abstractmethod


class Controller(metaclass=ABCMeta):
    """Controller struct"""
    @abstractmethod
    def action(self, x: List[float]) -> List[float]:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> List[float]:
        pass

    @abstractmethod
    def save(self, filedir: str=None):
        pass
