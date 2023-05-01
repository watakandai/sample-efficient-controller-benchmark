from abc import ABCMeta, abstractmethod
from ..countersampler.base import CounterExample
from ..controller.base import Controller


# class TrainerArguments(metaclass=ABCMeta):
#     """Arguments struct"""
#     pass


class Trainer(metaclass=ABCMeta):
    """Trainer samples a trajectory and updates the controller"""
    _controller: Controller = None

    @abstractmethod
    def train(self, counterexample: CounterExample) -> Controller:
        pass
