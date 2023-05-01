from .base import Trainer, TrainerArguments
from ..countersampler.base import CounterExample
from ..controller.rlcontroller import RLController


# class RLTrainerArguments(TrainerArguments):
#     """Arguments struct"""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


class RLTrainer(Trainer):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def train(self, counterexample: CounterExample) -> RLController:

        self.env.reset()
        s = counterexample.x

        done = False

        while not done:
            a = self._controller.action(s)
            ns, r, d, info = self.env.step(a)
            self._controller.update(s, a, r, d, ns, info)

        return self._controller
