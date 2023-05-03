import random
from abc import ABCMeta, abstractmethod
from typing import Dict, List
from ..verifier.base import VerifierResult
from .. import SimStatus


class CounterExample():
    """CounterExample struct"""
    x: List[float]
    def __init__(self, x):
        self.x = x


class CounterSampler(metaclass=ABCMeta):
    """CounterSample identifies and returns a CounterExample"""
    @abstractmethod
    def sample(self, result: VerifierResult) -> CounterExample:
        raise NotImplementedError()


class FirstXOfRandomTrajSampler(CounterSampler):
    def sample(self, result: VerifierResult) -> CounterExample:
        trajs = result.trajectories
        unsafeTraj = list(filter(lambda t: t["status"]!=SimStatus.SIM_TERMINATED, trajs))
        traj = random.choice(unsafeTraj)
        x = traj["X"][0]
        return CounterExample(x)


class RandomXOfRandomTrajSampler(CounterSampler):
    def sample(self, result: VerifierResult) -> CounterExample:
        trajs = result.trajectories
        unsafeTraj = list(filter(lambda t: t["status"]!=SimStatus.SIM_TERMINATED, trajs))
        traj = random.choice(unsafeTraj)
        x = random.choice(traj.X)
        return CounterExample(x)

class RandomX0TrajSampler(CounterSampler):
    def __init__(self, env):
        self.env = env

    def sample(self, result: VerifierResult) -> CounterExample:
        return CounterExample(self.env.randX0())
