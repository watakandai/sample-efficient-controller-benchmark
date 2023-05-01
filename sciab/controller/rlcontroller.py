from abc import ABCMeta, abstractmethod
from .base import Controller


class RLController(Controller):
    """Controller struct"""
    # TODO: Neural Network!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
