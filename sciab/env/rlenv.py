from typing import List, Tuple
from .base import Env, EnvArguments


class RLEnvArguments(EnvArguments):
    def __init__(self, system, **kwargs):
        super().__init__(**kwargs)
        self.system = system
        # TODO: Instiate an rl env


class RLEnv(Env):
    def __init__(self, args: RLEnvArguments):
        super().__init__(args)

    def sample(self, x: List[float]):
        raise NotImplementedError()

    def reset(self) -> List[float]:
        return self.args.env.reset()

    def step(self, a: List[float]) -> Tuple[List[float], float, bool, dict]:
        return self.args.env.step(a)
