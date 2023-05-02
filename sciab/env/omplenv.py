import os
import re
import shlex
import subprocess as sp
import numpy as np
import gymnasium as gym
from functools import reduce
from gymnasium.envs.registration import register
from typing import List, Tuple, Any, Dict, Union

from .base import Env, EnvArguments
from .. import SimStatus

def flatten(nestedList: List[List[float]]) -> List[float]:
    return reduce(lambda x, y: x + y, nestedList)

npToStr = lambda arr : " ".join(map(lambda x: f"{x:.2f}", arr))


class OMPLEnv(gym.Env):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float],
                       initUpperBound: List[float],
                       goalLowerBound: List[float],
                       goalUpperBound: List[float],
                       safeLowerBound: List[float],
                       safeUpperBound: List[float],
                       controlLowerBound: List[float],
                       controlUpperBound: List[float],
                       system: str,
                       sampleBinName: str="sampleOMPL",
                       stepBinName: str="stepOMPL",
                       outputFile: str="path.txt",
                       omplkwargs: Dict[str, str]={},
                       **kwargs):
        super(OMPLEnv, self).__init__(**kwargs)

        self.observation_space =  gym.spaces.Box(low=np.array(safeLowerBound),
                                    high=np.array(safeUpperBound),
                                    dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "action": gym.spaces.Discrete(1),
            "control": gym.spaces.Box(low=np.array(controlLowerBound),
                                      high=np.array(controlUpperBound),
                                      dtype=np.float32)})

        # Required Arguments
        self.pathToExecutable = pathToExecutable
        self.initLowerBound = np.array(initLowerBound)
        self.initUpperBound = np.array(initUpperBound)
        self.goalLowerBound = np.array(goalLowerBound)
        self.goalUpperBound = np.array(goalUpperBound)
        self.safeLowerBound = np.array(safeLowerBound)
        self.safeUpperBound = np.array(safeUpperBound)
        self.controlLowerBound = np.array(controlLowerBound)
        self.controlUpperBound = np.array(controlUpperBound)

        self.system = system
        self.sampleBinName = sampleBinName
        self.stepBinName = stepBinName
        self.outputFile = outputFile
        self.kwargs = omplkwargs
        self.mode = 1       # always start from 1
        self.state = None   # state is None at the beginning

    def runBinary(self, pathToExecutable: str, **kwargs):
        flatten = lambda nestedList: reduce(lambda x, y: x + y, nestedList)
        args = flatten([[f"--{k}={v}"] for k, v in kwargs.items()])

        commandList = [pathToExecutable] + args
        args = shlex.split(" ".join(commandList))

        p = sp.run(args, stdout=sp.PIPE, stderr=sp.PIPE)

        # if p.returncode != 0:
        #     raise Exception(str(p.stderr.decode()))

        return p.stdout.decode()

    def sample(self, x: List[float]):

        x0 = self.randX0()
        kwargs = {"system":            self.system,
                  "start":             npToStr(x0),
                  "goalLowerBound":    npToStr(self.goalLowerBound),
                  "goalUpperBound":    npToStr(self.goalUpperBound),
                  "lowerBound":        npToStr(self.safeLowerBound),
                  "upperBound":        npToStr(self.safeUpperBound),
                  "controlLowerBound": npToStr(self.controlLowerBound),
                  "controlUpperBound": npToStr(self.controlUpperBound)}
        kwargs.update(self.kwargs)

        pathToExecutable = os.path.join(self.pathToExecutable, self.sampleBinName)
        output = self.runBinary(pathToExecutable, **kwargs)

        print("="*100)
        print(output)
        print("="*100)

        if "Found a solution" in output: # and\
            # "Solution is approximate" not in output:
            data = np.loadtxt(self.outputFile)
            nx = len(self.initLowerBound)
            nu = len(self.controlLowerBound)
            X = data[:, :nx]
            U = data[:, nx:nx+nu]
            Dt = data[:, -1]

            return X, U, Dt

        return [], [], []

    def inTerminal(self, x):
        return all(self.goalLowerBound <= x) and all(x <= self.goalUpperBound)

    def outOfBound(self, x):
        return any(x < self.safeLowerBound) or any(self.safeUpperBound < x)

    def randX0(self) -> List[float]:
        numState = len(self.initLowerBound)
        return self.initLowerBound + self.np_random.random(numState) * (self.initUpperBound - self.initLowerBound)

    def reset(self, seed=None, options=None) -> List[float]:
        super().reset(seed=seed)
        state = self.randX0()
        while self.inTerminal(state) or self.outOfBound(state):
            state = self.randX0()
        self.state = state
        return self._get_obs()

    def _get_obs(self):
        # return {"mode": self.mode, "state": self.state}
        return self.state

    def _get_info(self):
        return {"status": SimStatus.SIM_INFEASIBLE}

    def step(self, action: Union[int, List[float]]) -> Tuple[List[float], float, bool, dict]:

        if isinstance(action, int):
            a = action
            u = np.zeros(len(self.controlLowerBound))
        elif isinstance(action, List) or isinstance(action, np.ndarray):
            u = action
            a = 1
        else:
            raise Exception("Invalid action type")

        if self.state is None:
            raise Exception("You must call reset() before calling step()")

        pathToExecutable = os.path.join(self.pathToExecutable, self.stepBinName)
        kwargs = {"system":            self.system,
                  "state":             npToStr(self.state),
                  "mode":              str(self.mode),
                  "action":            str(a),
                  "control":           npToStr(u)}
        output = self.runBinary(pathToExecutable, **kwargs)

        mode_pattern = r"q=(\d+)" # This pattern matches "q=int"
        # x_pattern = r'x=([\d\.\-\s]+)' # This pattern matches "x=double1 double2 ..."
        x_pattern = r'x=([\d\.\-e ]+)'
        mode_match = re.search(mode_pattern, output) # Search for the mode pattern in the string.
        x_match = re.search(x_pattern, output) # Search for the x pattern in the string.

        if mode_match and x_match:
            self.mode = int(mode_match.group(1)) # Convert the matched mode to an integer.
            self.state = [float(s) for s in x_match.group(1).split()] # Convert the matched groups to a list of floats.
        else:
            raise Exception("No match found.")

        done = False
        truncated = False
        info = self._get_info()
        info["output"] = output

        if self.inTerminal(self.state):
            info["status"] = SimStatus.SIM_TERMINATED
            done = True
        elif self.outOfBound(self.state):
            info["status"] = SimStatus.SIM_UNSAFE
            truncated = True
            done = True

        reward = int(done)
        return self._get_obs(), reward, done, truncated, info


class DubinsCar(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.95, -0.05],
                       initUpperBound: List[float]=[ 0.05, 1.05,  0.05],
                       goalLowerBound: List[float]=[1.0, -0.1, -0.1],
                       goalUpperBound: List[float]=[2.0,  0.1,  0.1],
                       safeLowerBound: List[float]=[-0.5, -0.5, -np.pi],
                       safeUpperBound: List[float]=[ 2.0,  2.0,  np.pi],
                       controlLowerBound: List[float]=[-0.1],
                       controlUpperBound: List[float]=[ 0.1],
                       **kwargs):
        super().__init__(pathToExecutable,
                         initLowerBound,
                         initUpperBound,
                         goalLowerBound,
                         goalUpperBound,
                         safeLowerBound,
                         safeUpperBound,
                         controlLowerBound,
                         controlUpperBound,
                         system="DubinsCar",
                         **kwargs)


class DubinsCarWithAcceleration(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.95, -0.05],
                       initUpperBound: List[float]=[ 0.05, 1.05,  0.05],
                       goalLowerBound: List[float]=[1.0, -0.1, -0.1],
                       goalUpperBound: List[float]=[2.0,  0.1,  0.1],
                       safeLowerBound: List[float]=[-0.5, -0.5, -np.pi],
                       safeUpperBound: List[float]=[ 2.0,  2.0,  np.pi],
                       controlLowerBound: List[float]=[-0.1],
                       controlUpperBound: List[float]=[ 0.1],
                       **kwargs):
        super().__init__(pathToExecutable,
                         initLowerBound,
                         initUpperBound,
                         goalLowerBound,
                         goalUpperBound,
                         safeLowerBound,
                         safeUpperBound,
                         controlLowerBound,
                         controlUpperBound,
                         system="DubinsCarWithAcceleration",
                         **kwargs)


class Unicycle(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.95, -0.05],
                       initUpperBound: List[float]=[ 0.05, 1.05,  0.05],
                       goalLowerBound: List[float]=[1.0, -0.1, -0.1],
                       goalUpperBound: List[float]=[2.0,  0.1,  0.1],
                       safeLowerBound: List[float]=[-0.5, -0.5, -np.pi],
                       safeUpperBound: List[float]=[ 2.0,  2.0,  np.pi],
                       controlLowerBound: List[float]=[-0.1],
                       controlUpperBound: List[float]=[ 0.1],
                       **kwargs):
        super().__init__(pathToExecutable,
                         initLowerBound,
                         initUpperBound,
                         goalLowerBound,
                         goalUpperBound,
                         safeLowerBound,
                         safeUpperBound,
                         controlLowerBound,
                         controlUpperBound,
                         system="Unicycle",
                         **kwargs)


register(
     id="ompl/DubinsCar-v0",
     entry_point="sciab.env.omplenv:DubinsCar",
     max_episode_steps=50,
)
