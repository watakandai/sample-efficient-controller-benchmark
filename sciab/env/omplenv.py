import os
import re
import copy
import shlex
import subprocess as sp
import numpy as np
import gymnasium as gym
from functools import reduce
from gymnasium.envs.registration import register
from typing import List, Tuple, Union
from .. import SimStatus
gym.logger.set_level(40)

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
                       dt: float,
                       system: str,
                       sampleBinName: str="sampleOMPL",
                       stepBinName: str="stepOMPL",
                       outputFile: str="path.txt",
                       objective: str="PathLength",
                       runTime: float=5,
                       goalBias: float=0.05,
                       samplingBias: float=0.2,
                       pruningRadius: float=0.05,
                       selectionRadius: float=0.02,
                       propagationStepSize: float=0.2,
                       controlDurationBound: List[float]=[1, 10],
                       **kwargs):
        super(OMPLEnv, self).__init__(**kwargs)

        self.observation_space =  gym.spaces.Box(low=np.array(safeLowerBound),
                                    high=np.array(safeUpperBound),
                                    dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array(controlLowerBound),
            high=np.array(controlUpperBound),
            dtype=np.float32)

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
        self.objective = objective
        self.runTime = runTime
        self.goalBias = goalBias
        self.samplingBias = samplingBias
        self.pruningRadius = pruningRadius
        self.selectionRadius = selectionRadius
        self.propagationStepSize = propagationStepSize
        self.controlDurationBound = controlDurationBound

        self.dt = dt
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

    def sampleTrajectory(self, x: List[float]):

        x0 = self.randX0()
        kwargs = {"system":                 self.system,
                  "start":                  npToStr(x0),
                  "goalLowerBound":         npToStr(self.goalLowerBound),
                  "goalUpperBound":         npToStr(self.goalUpperBound),
                  "lowerBound":             npToStr(self.safeLowerBound),
                  "upperBound":             npToStr(self.safeUpperBound),
                  "controlLowerBound":      npToStr(self.controlLowerBound),
                  "controlUpperBound":      npToStr(self.controlUpperBound),
                  "propagationStepSize":    str(self.dt)}
        pathToExecutable = os.path.join(self.pathToExecutable, self.sampleBinName)


        output = self.runBinary(pathToExecutable, **kwargs)

        print("="*100)
        print(output)
        print("="*100)

        if "Found a solution" in output and\
            "Solution is approximate" not in output:
            data = np.loadtxt(self.outputFile)
            nx = len(self.initLowerBound)
            nu = len(self.controlLowerBound)
            X = data[:, :nx]
            A = data[1:, nx:nx+1]
            U = data[1:, nx+1:nx+1+nu]
            Dt = data[1:, -1]

            return X, A, U, Dt

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
        return copy.deepcopy(self.state)

    def _get_info(self):
        return {"status": SimStatus.SIM_INFEASIBLE}

    def stepCall(self, action: Union[int, List[float]]) -> Tuple[List[float], float, bool, dict]:

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
        kwargs = {"system":                 self.system,
                  "state":                  npToStr(self.state),
                  "mode":                   str(self.mode),
                  "action":                 str(a),
                  "control":                npToStr(u),
                  "propagationStepSize":    self.dt}
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

        self.dynamics(u, a)

        done = False
        truncated = False
        info = self._get_info()

        if self.inTerminal(self.state):
            info["status"] = SimStatus.SIM_TERMINATED
            done = True
        elif self.outOfBound(self.state):
            info["status"] = SimStatus.SIM_UNSAFE
            truncated = True
            done = True

        reward = int(done)
        return self._get_obs(), reward, done, truncated, info

    def dynamics(self, u, a):
        raise NotImplementedError()


class Car2D(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-np.pi/3, -0.3],
                       initUpperBound: List[float]=[ np.pi/3,  0.3],
                       goalLowerBound: List[float]=[-np.pi/12, -0.3],
                       goalUpperBound: List[float]=[ np.pi/12,  0.3],
                       safeLowerBound: List[float]=[-np.pi/2, -1.0],
                       safeUpperBound: List[float]=[ np.pi/2,  1.0],
                       controlLowerBound: List[float]=[-1],
                       controlUpperBound: List[float]=[ 1],
                       dt=0.2,
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
                         system="Car2D",
                         dt=dt,
                         **kwargs)

    def dynamics(self, control: List[float], action: int):
        velocity = 0.05
        theta = self.state[0]
        self.state[0] += control[0]*self.dt
        self.state[1] += velocity*theta*self.dt


class DubinsCar(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.95, -0.05],
                       initUpperBound: List[float]=[ 0.05, 1.05,  0.05],
                       goalLowerBound: List[float]=[1.0, -0.1, -0.2],
                       goalUpperBound: List[float]=[3.0,  0.1,  0.2],
                       safeLowerBound: List[float]=[-0.5, -0.5, -np.pi],
                       safeUpperBound: List[float]=[ 3.0,  2.0,  np.pi],
                       controlLowerBound: List[float]=[-0.1],
                       controlUpperBound: List[float]=[ 0.1],
                       dt=0.2,
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
                         dt=dt,
                         **kwargs)

    def dynamics(self, control: List[float], action: int):
        velocity = 0.1
        omega = control[0]
        self.state[0] += velocity*np.cos(self.state[2])*self.dt
        self.state[1] += velocity*np.sin(self.state[2])*self.dt
        self.state[2] += omega*self.dt


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

    def dynamics(self, u, a):
        return super().dynamics(u, a)


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


register(id="ompl/Car2D-v0",
         entry_point="sciab.env.omplenv:Car2D",
         max_episode_steps=50)
register(id="ompl/DubinsCar-v0",
         entry_point="sciab.env.omplenv:DubinsCar",
         max_episode_steps=50)
register(id="ompl/DubinsCarWithAcceleration-v0",
         entry_point="sciab.env.omplenv:DubinsCarWithAcceleration",
         max_episode_steps=50)
register(id="ompl/Unicycle-v0",
         entry_point="sciab.env.omplenv:Unicycle",
         max_episode_steps=50)
register(id="ompl/Unicycle-v0",
         entry_point="sciab.env.omplenv:Unicycle",
         max_episode_steps=50)
register(id="ompl/Unicycle-v0",
         entry_point="sciab.env.omplenv:Unicycle",
         max_episode_steps=50)
register(id="ompl/InvertedPendulum-v0",
         entry_point="sciab.env.omplenv:InvertedPendulum",
         max_episode_steps=50)
register(id="ompl/DoubleInvertedPendulum-v0",
         entry_point="sciab.env.omplenv:DoubleInvertedPendulum",
         max_episode_steps=50)
register(id="ompl/Drone2D-v0",
         entry_point="sciab.env.omplenv:Drone2D",
         max_episode_steps=50)
register(id="ompl/Drone3D-v0",
         entry_point="sciab.env.omplenv:Drone3D",
         max_episode_steps=50)
register(id="ompl/CaltechDuctedFan-v0",
         entry_point="sciab.env.omplenv:CaltechDuctedFan",
         max_episode_steps=50)
