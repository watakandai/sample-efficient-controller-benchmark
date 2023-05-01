import random
from abc import ABCMeta, abstractmethod
from typing import Dict, List
from ..verifier.base import VerifierResult
from .. import SimStatus


class CounterExample(metaclass=ABCMeta):
    """CounterExample struct"""
    x: List[float]


class CounterSampler(metaclass=ABCMeta):
    """CounterSample identifies and returns a CounterExample"""
    @abstractmethod
    def sample(self, result: VerifierResult) -> CounterExample:
        raise NotImplementedError()


"""CounterExample struct"""
class BaseCounterExample(CounterExample): pass


class FirstXOfRandomTrajSampler(CounterSampler):
    def sample(self, result: VerifierResult) -> CounterExample:
        trajs = result.trajectories
        unsafeTraj = filter(lambda t: t["status"]!=SimStatus.SIM_TERMINATED, trajs)
        traj = random.choice(unsafeTraj)
        x = traj.X[0]
        return BaseCounterExample(x)


class RandomXOfRandomTrajSampler(CounterSampler):
    def sample(self, result: VerifierResult) -> CounterExample:
        trajs = result.trajectories
        unsafeTraj = filter(lambda t: t["status"]!=SimStatus.SIM_TERMINATED, trajs)
        traj = random.choice(unsafeTraj)
        x = random.choice(traj.X)
        return BaseCounterExample(x)
