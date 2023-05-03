from .base import Trainer
from ..countersampler.base import CounterExample
from ..controller.voronoicontroller import VoronoiController
from ..env.omplenv import OMPLEnv



class VoronoiTrainer(Trainer):
    def __init__(self, env: OMPLEnv):
        super().__init__(env)
        self.controller = VoronoiController()

    def train(self, counterexample: CounterExample) -> VoronoiController:
        X, A, U, Dt = self.env.sampleTrajectory(counterexample.x)
        self.controller.update(X, U, Dt)
        return self.controller
