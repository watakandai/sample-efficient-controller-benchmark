from abc import ABCMeta, abstractmethod
from ..countersampler.base import CounterExample
from ..controller.base import Controller
from ..env.omplenv import OMPLEnv


# class TrainerArguments(metaclass=ABCMeta):
#     """Arguments struct"""
#     pass


class Trainer(metaclass=ABCMeta):
    """Trainer samples a trajectory and updates the controller"""
    controller: Controller = None
    env: OMPLEnv = None

    @abstractmethod
    def __init__(self, env: OMPLEnv) -> Controller:
        self.env = env


    @abstractmethod
    def train(self, counterexample: CounterExample) -> Controller:
        pass
