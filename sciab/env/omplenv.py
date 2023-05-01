import subprocess
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple
from .base import Env, EnvArguments
from .. import SimStatus


class OMPLEnvArguments(EnvArguments):
    def __init__(self, system: str,
                       pathToExecutable: str,
                       # Make the followings optional
                       start: List[float]=None,
                       goalLowerBound: List[float]=None,
                       goalUpperBound: List[float]=None,
                       safeLowerBound: List[float]=None,
                       safeUpperBound: List[float]=None,
                       controlLowerBound: List[float]=None,
                       controlUpperBound: List[float]=None,
                       runTime: int=1,
                       selectionRadius: float=0.6,
                       pruningRadius: float=0.08,
                       propagationStepSize: float=0.1,
                       controlDurationBound: List[int]=[1, 10],
                       planner: str="SST",
                       objective: str="PathLengthObjWithCostToGo",
                       file: str="path.txt",
                       **kwargs):
        # Required Arguments
        self.system = system
        self.pathToExecutable = pathToExecutable
        self.start = start
        self.goalLowerBound = goalLowerBound
        self.goalUpperBound = goalUpperBound
        self.safeLowerBound = safeLowerBound
        self.safeUpperBound = safeUpperBound
        self.controlLowerBound = controlLowerBound
        self.controlUpperBound = controlUpperBound
        # Optional Arguments
        self.runTime = runTime
        self.selectionRadius = selectionRadius
        self.pruningRadius = pruningRadius
        self.propagationStepSize = propagationStepSize
        self.controlDurationBound = controlDurationBound
        self.planner = planner
        self.objective = objective
        self.file = file

        super().__init__(**kwargs)


class OMPLEnvWrapper(gym.Wrapper):
    pass


class OMPLEnv(Env):
    def __init__(self, args: OMPLEnvArguments):
        super().__init__(args)

    def sample(self, x: List[float]):
        pathToExecutable = self.args.pathToExecutable
        args = ["--system",                 self.args.system,
                "--start",                  self.args.start,
                "--goal",                   self.args.goal,
                "--goalLowerBound",         self.args.goalLowerBound,
                "--goalUpperBound",         self.args.goalUpperBound,
                "--safeLowerBound",         self.args.safeLowerBound,
                "--safeUpperBound",         self.args.safeUpperBound,
                "--controlLowerBound",      self.args.controlLowerBound,
                "--controlUpperBound",      self.args.controlUpperBound,
                "--runTime",                self.args.runTime,
                "--goalBias",               self.args.goalBias,
                "--selectionRadius",        self.args.selectionRadius,
                "--pruningRadius",          self.args.pruningRadius,
                "--propagationStepSize",    self.args.propagationStepSize,
                "--controlDurationBound",   self.args.controlDurationBound,
                "--planner",                self.args.planner,
                "--objective",              self.args.objective,
                "--file",                   self.args.file]

        commandList = [pathToExecutable] + args
        process = subprocess.Popen(commandList, stdout=subprocess.PIPE)
        output, error = process.communicate()

        print("="*100)
        print(output)
        print("="*100)

        if "Found a solution" in output and\
            "Solution is approximate" not in output:
            data = np.loadtxt(self.args.file)
            nx = len(self.args.safeLowerBound)
            nu = len(self.args.controlLowerBound)
            X = data[:, :nx]
            U = data[:, nx:nx+nu]
            Dt = data[:, -1]

            return X, U, Dt

        return [], [], []

    def inTerminal(self, x):
        return all(self.termSet.lb <= x) and all(x <= self.termSet.ub)

    def outOfBound(self, x):
        return any(x < self.workspace.lb) or any(self.workspace.ub < x)

    def maxReached(self):
        return self.currStep >= self.numStep

    def reset(self) -> List[float]:
        # TODO:
        numState = 1
        x = self.initSet.lb + np.random.rand(numState) * (self.initSet.ub - self.initSet.lb)
        while self.inTerminal(x):
            x = self.initSet.lb + np.random.rand(numState) * (self.initSet.ub - self.initSet.lb)
        return x

    def step(self, a: List[float]) -> Tuple[List[float], float, bool, dict]:

        # TODO:
        next_x = self.dynamics(self.x, a)

        reward = 0
        done = False
        info = {}

        if self.inTerminal(next_x):
            info["status"] = SimStatus.SIM_TERMINATED
            done = True
        elif self.outOfBound(next_x):
            info["status"] = SimStatus.SIM_UNSAFE
            done = True
        elif self.maxReached():
            info["status"] = SimStatus.SIM_MAX_ITER_REACHED
            done = True

        return next_x, reward, done, info


from gymnasium.envs.registration import register



register(
     id="ompl/Drone-v0",
     entry_point="sciab.env.omplenv:DroneEnv",
     max_episode_steps=50,
)
