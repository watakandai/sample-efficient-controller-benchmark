from .base import Trainer
from ..countersampler.base import CounterExample
from ..controller.voronoicontroller import VoronoiController
from ..env.omplenv import OMPLEnv


# class VoronoiTrainerArguments(TrainerArguments):
#     """Arguments struct"""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


class VoronoiTrainer(Trainer):
    def __init__(self, env: OMPLEnv):
        super().__init__()
        self.env = env

    def train(self, counterexample: CounterExample) -> VoronoiController:
        X, U, Dt = self.env.sample(counterexample.x)
        self._controller.update(X, U, Dt)
        return self._controller
