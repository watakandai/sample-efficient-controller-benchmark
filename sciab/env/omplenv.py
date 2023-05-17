import re
import os
import re
import copy
import shlex
import logging
import numpy as np
import subprocess as sp
import gymnasium as gym
from pathlib import Path
from scipy.integrate import odeint, solve_ivp
from functools import reduce
from multiprocessing.pool import ThreadPool
from gymnasium.envs.registration import register
from typing import List, Tuple, Union, Dict
from .. import SimStatus
gym.logger.set_level(40)

def rpy2rotmat(rpy):
    cos_roll = np.cos(rpy[0])
    sin_roll = np.sin(rpy[0])
    cos_pitch = np.cos(rpy[1])
    sin_pitch = np.sin(rpy[1])
    cos_yaw = np.cos(rpy[2])
    sin_yaw = np.sin(rpy[2])
    R_roll = np.array([[1., 0, 0], [0, cos_roll, -sin_roll],
                        [0, sin_roll, cos_roll]])
    R_pitch = np.array([[cos_pitch, 0, sin_pitch], [0, 1., 0],
                        [-sin_pitch, 0, cos_pitch]])
    R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0],
                        [0, 0, 1.]])
    return R_yaw @ R_pitch @ R_roll

npToStr = lambda arr : " ".join(map(lambda x: f"{x}", arr))
angle_normalize = lambda x :((x + np.pi) % (2 * np.pi)) - np.pi


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
                       stateWeight: List[float],
                       controlWeight: List[float],
                       dt: float,
                       system: str,
                       outputFile: str="tmp/path.txt",
                       objective: str="PathLength",
                       runTime: float=10,
                       goalBias: float=0.05,
                       samplingBias: float=0.2,
                       pruningRadius: float=0.1,
                       selectionRadius: float=0.2,
                       controlDurationBound: List[int]=[1, 20],
                       sampleBinName: str="sampleOMPL",
                       max_episode_steps: int=30,
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
        self.goalCenter = (self.goalUpperBound + self.goalLowerBound) / 2.0
        self.safeLowerBound = np.array(safeLowerBound)
        self.safeUpperBound = np.array(safeUpperBound)
        self.controlLowerBound = np.array(controlLowerBound)
        self.controlUpperBound = np.array(controlUpperBound)
        self.stateWeight = np.array(stateWeight)
        self.controlWeight = np.array(controlWeight)

        self.system = system
        self.outputFile = outputFile
        self.objective = objective
        self.runTime = runTime
        self.goalBias = goalBias
        self.samplingBias = samplingBias
        self.pruningRadius = pruningRadius
        self.selectionRadius = selectionRadius
        self.controlDurationBound = controlDurationBound

        self.sampleBinName = sampleBinName
        self._max_episode_steps = max_episode_steps

        self.dt = dt
        self.state = None   # state is None at the beginning
        self._reset()

    def runBinary(self, pathToExecutable: str, **kwargs):
        flatten = lambda nestedList: reduce(lambda x, y: x + y, nestedList)
        args = flatten([[f"--{k}={v}"] for k, v in kwargs.items()])

        commandList = [pathToExecutable] + args
        # print(commandList)
        args = shlex.split(" ".join(commandList))
        p = sp.run(args, stdout=sp.PIPE, stderr=sp.PIPE)

        # if p.returncode != 0:
        #     raise Exception(str(p.stderr.decode()))

        return p.stdout.decode()

    def sampleTrajectory(self, x: List[float], returnCost: bool=False, ind: int=None):

        p = Path(self.outputFile)
        p.parent.mkdir(parents=True, exist_ok=True)
        file = p.parent.joinpath(p.stem + str(ind) + f.suffix) if ind else p

        kwargs = {"system":                 self.system,
                  "start":                  npToStr(x),
                  "goalLowerBound":         npToStr(self.goalLowerBound),
                  "goalUpperBound":         npToStr(self.goalUpperBound),
                  "lowerBound":             npToStr(self.safeLowerBound),
                  "upperBound":             npToStr(self.safeUpperBound),
                  "controlLowerBound":      npToStr(self.controlLowerBound),
                  "controlUpperBound":      npToStr(self.controlUpperBound),
                  "controlDurationBound":   npToStr(self.controlDurationBound),
                  "stateWeight":            npToStr(self.stateWeight),
                  "controlWeight":          npToStr(self.controlWeight),
                  "propagationStepSize":    str(self.dt),
                  "runTime":                str(self.runTime),
                  "goalBias":               str(self.goalBias),
                  "samplingBias":           str(self.samplingBias),
                  "pruningRadius":          str(self.pruningRadius),
                  "selectionRadius":        str(self.selectionRadius),
                  "file":                   file}
        pathToExecutable = os.path.join(self.pathToExecutable, self.sampleBinName)

        output = self.runBinary(pathToExecutable, **kwargs)
        logging.debug(output)
        # print(output)

        if "Found a solution" in output and\
            "Solution is approximate" not in output:
        # if "Found a solution" in output:
            pattern = r"Found solution with cost (\d+\.\d+)"
            costs = re.findall(pattern, output)

            data = np.loadtxt(file)
            nx = len(self.initLowerBound)
            nu = len(self.controlLowerBound)
            X = data[:, :nx]
            A = data[1:, nx:nx+1]
            U = data[1:, nx+1:nx+1+nu]
            Dt = data[1:, -1]

            if returnCost:
                return X, A, U, Dt, min(costs)
            return X, A, U, Dt

        if returnCost:
            return [], [], [], [], np.inf
        return [], [], [], []

    def sampleTrajectories(self, x: List[float], numProcess: int=16, returnCost:bool=False):
        t = ThreadPool(processes=numProcess)
        f = lambda i : self.sampleTrajectory(x, returnCost=returnCost, ind=i)
        mapTrajectories = t.map(f, range(numProcess))
        t.close()
        return list(mapTrajectories)

    def sampleBestCostTrajectory(self, x: List[float], numProcess: int=16):
        trajectories = self.sampleTrajectories(x, numProcess, returnCost=True)
        costs = list(map(lambda t: t[4], trajectories))
        bestTrajectory = trajectories[np.argmin(costs)]
        return bestTrajectory[:4]

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
        c = 0
        while self.inTerminal(state) or self.outOfBound(state):
            state = self.randX0()
            c += 1
            if c > 100:
                raise Exception("Cannot find a valid initial state")

        assert(all(self.initLowerBound <= state) and all(state <= self.initUpperBound))

        self.state = state
        self._reset()
        return self._get_obs()

    def _reset(self):
        self.reward = -np.inf
        self.truncated = False
        self.done = False
        self.info = {}
        self.numStep = 0

    def _get_obs(self):
        return copy.deepcopy(self.state)

    def _get_reward(self):
        if self.truncated:
            return 0
        else:
            if self.done:
                return 1
        return -np.linalg.norm(self.goalCenter - self.state)

    def _get_done_truncated_info(self):

        done = False
        truncated = False
        info = {"status": SimStatus.SIM_INFEASIBLE}

        if self.inTerminal(self.state):
            info["status"] = SimStatus.SIM_TERMINATED
            done = True
        elif self.outOfBound(self.state):
            info["status"] = SimStatus.SIM_UNSAFE
            truncated = True

        if self.numStep >= self._max_episode_steps:
            truncated = True
            done = True

        self.done = done
        self.truncated = truncated
        self.info = info
        return done, truncated, info

    def step(self, action: Union[int, List[float], Dict]) -> Tuple[List[float], float, bool, dict]:

        duration = self.dt
        if isinstance(action, int):
            a = action
            u = np.zeros(len(self.controlLowerBound))
        elif isinstance(action, List) or isinstance(action, np.ndarray):
            u = action
            a = 1
        elif isinstance(action, Dict):
            u = action["control"]
            # a = action["action"]
            a = 1
            duration = action["dt"]
        else:
            raise Exception("Invalid action type")

        if self.state is None:
            raise Exception("You must call reset() before calling step()")

        if not self.truncated:
            t = np.arange(0, duration+self.dt, self.dt)
            num_sol = solve_ivp(self.dynamics, [0, duration+self.dt], self.state, method="RK45", args=(u, a))
            Xs = num_sol.y
            self.state = Xs[:, -1].T

        self.numStep += 1
        done, truncated, info = self._get_done_truncated_info()

        return self._get_obs(), self._get_reward(), done, truncated, info

    def dynamics(self, t, x, u, a):
        raise NotImplementedError()


"""X=[theta, y], U=[omega]"""
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
                       stateWeight: List[float]=[0.5, 1],
                       controlWeight: List[float]=[1],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Car2D",
                         **kwargs)

    def dynamics(self, t, x, control: List[float], action: int):
        velocity = 0.05
        theta = x[0]
        dxdt = np.zeros(x.shape)
        dxdt[0] = control[0]
        dxdt[1] = velocity * theta
        return dxdt


"""X=[x,y,θ], U=[\omega]"""
class DubinsCar(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, -0.05, 5*np.pi/12],
                       initUpperBound: List[float]=[ 0.05,  0.05, 6*np.pi/12],
                       goalLowerBound: List[float]=[0.3, -0.1, -0.5],
                       goalUpperBound: List[float]=[0.4,  0.1,  0.5],
                       safeLowerBound: List[float]=[-0.5, -0.3, -np.pi],
                       safeUpperBound: List[float]=[ 1.0,  0.3,  np.pi],
                       controlLowerBound: List[float]=[-1.0],
                       controlUpperBound: List[float]=[ 1.0],
                       stateWeight: List[float]=[1, 1, 0.5],
                       controlWeight: List[float]=[1],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "DubinsCar",
                         **kwargs)

    def dynamics(self, t, x, control: List[float], action: int):
        velocity = 0.1
        omega = control[0]
        dxdt = np.zeros(len(x))
        dxdt[0] = velocity * np.cos(x[2])
        dxdt[1] = velocity * np.sin(x[2])
        dxdt[2] = omega
        return dxdt

    def _get_reward(self):
        if not self.truncated and self.done:
            return 1
        # Only take x, y
        dx = self.goalCenter[0] - self.state[0]
        dy = self.goalCenter[1] - self.state[1]
        dth = self.goalCenter[2] - self.state[2]
        return -10*np.sqrt(dx*dx + dy*dy + 0.5*dth*dth)


"""X=[x,y,θ,v], U=[\omega, a]"""
class DubinsCarWithAcceleration(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, -0.05, 5*np.pi/12, -0.05],
                       initUpperBound: List[float]=[ 0.05,  0.05, 6*np.pi/12,  0.05,],
                       goalLowerBound: List[float]=[0.3, 0.0, -np.pi, -3],
                       goalUpperBound: List[float]=[0.5, 0.1,  np.pi,  3],
                       safeLowerBound: List[float]=[-0.1, -0.1, -np.pi, -3.0],
                       safeUpperBound: List[float]=[ 0.6,  0.6,  np.pi,  3.0],
                       controlLowerBound: List[float]=[-1.0, -0.1],
                       controlUpperBound: List[float]=[ 1.0,  0.1],
                       stateWeight: List[float]=[1, 1, 0, 0],
                       controlWeight: List[float]=[2, 0.5],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "DubinsCarWithAcceleration",
                         **kwargs)

    def dynamics(self, t, x, control: List[float], action: int):
        omega = control[0]
        acceleration = control[1]
        theta = x[2]
        velocity = x[3]
        dxdt = np.zeros(len(x))
        dxdt[0] = velocity * np.cos(theta)
        dxdt[1] = velocity * np.sin(theta)
        dxdt[2] = omega
        dxdt[3] = acceleration
        return dxdt

    def _get_reward(self):
        if not self.truncated and self.done:
            return 1
        dx = self.goalCenter[0] - self.state[0]
        dy = self.goalCenter[1] - self.state[1]
        dth = self.goalCenter[2] - self.state[2]
        dv = self.goalCenter[3] - self.state[3]

        return -10*np.sqrt(2*dx*dx + 2*dy*dy + 0.5*dth*dth + 0.1*dv*dv)
        # return sqrt(dx*dx + dy*dy)


"""X=[x,y,v], U=[\theta, a]"""
class Unicycle(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, -0.05, -0.5],
                       initUpperBound: List[float]=[ 0.05,  0.05,  0.5],
                       goalLowerBound: List[float]=[0.3, 0.3, -0.5],
                       goalUpperBound: List[float]=[0.5, 0.5,  0.5],
                       safeLowerBound: List[float]=[-0.1, -0.1, -1.0],
                       safeUpperBound: List[float]=[ 0.6,  0.6,  1.0],
                       controlLowerBound: List[float]=[-1.0, -1.0],
                       controlUpperBound: List[float]=[ 1.0,  1.0],
                       stateWeight: List[float]=[1, 1, 0],
                       controlWeight: List[float]=[0.5, 2.],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Unicycle",
                         **kwargs)

    def dynamics(self, t, x, control: List[float], action: int):
        theta = control[0]
        acceleration = control[1]
        dxdt = np.zeros(len(x))
        dxdt[0] = x[2]*np.cos(theta)
        dxdt[1] = x[2]*np.sin(theta)
        dxdt[2] = acceleration
        return dxdt

    def _get_reward(self):
        if not self.truncated and self.done:
            return 1
        dx = self.goalCenter[0] - self.state[0]
        dy = self.goalCenter[1] - self.state[1]
        dv = self.goalCenter[2] - self.state[2]
        return -10*np.sqrt(dx*dx + dy*dy + 0.1*dv*dv)
        # return -np.sqrt(dx*dx + dy*dy)
        # return -np.linalg.norm(self.goalCenter - self.state)
        # return -1


"""X=[x,y,v,phi,theta], U=[acceleration, handleAngle]"""
class UnicycleWithConstraint(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.95, -0.1, -0.1, -0.1],
                       initUpperBound: List[float]=[ 0.05, 1.05,  0.1, -0.1, -0.1],
                       goalLowerBound: List[float]=[1.4, 1.4, -0.5, -3.14, -3.14],
                       goalUpperBound: List[float]=[1.6, 1.6,  0.5,  3.14, 3.14],
                       safeLowerBound: List[float]=[-0.5, -0.5, -1.0, -1.0, -3.14],
                       safeUpperBound: List[float]=[ 2.0,  2.5,  1.0, 1.0, 3.14],
                       controlLowerBound: List[float]=[-1.0, -1.0],
                       controlUpperBound: List[float]=[ 1.0,  1.0],
                       stateWeight: List[float]=[1, 1, 0, 0, 0],
                       controlWeight: List[float]=[4, 0.5],
                       dt=0.1,
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "UnicycleWithConstraint",
                         **kwargs)
        self.carLength = 0.2

    def dynamics(self, t, x, control: List[float], action : int):

        # // state params
        v = x[2]
        phi = x[3]
        theta = x[4]

        theta = theta % (2.0*np.pi)
        if theta < -np.pi:
            theta += 2.0*np.pi
        elif theta >= np.pi:
            theta -= 2.0*np.pi

        # // Zero out qdot
        dxdt = np.zeros(len(x))

        # // calculate qdot
        dxdt[0] = v * np.cos(theta)
        dxdt[1] = v * np.sin(theta)
        dxdt[2] = control[0]
        dxdt[3] = control[1]
        dxdt[4] = (v / self.carLength) * np.tan(phi)

        return dxdt

    def _get_reward(self):
        if not self.truncated and self.done:
            return 1
        dx = self.goalCenter[0] - self.state[0]
        dy = self.goalCenter[1] - self.state[1]
        dv = self.goalCenter[2] - self.state[2]
        return -10*np.sqrt(dx*dx + dy*dy + dv*dv)
        # return -10*np.linalg.norm(self.goalCenter - self.state)


"""X=[x,dx,theta,dtheta], U=[force]"""
class CartPole(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, -0.05, 0.25, -0.1],
                       initUpperBound: List[float]=[ 0.05,  0.05, 0.30,  0.1],
                       goalLowerBound: List[float]=[-2.4, -0.3, -0.05, -0.05],
                       goalUpperBound: List[float]=[ 2.4,  0.3,  0.05,  0.05],
                       safeLowerBound: List[float]=[-4.8, -10, -0.4, -10],
                       safeUpperBound: List[float]=[ 4.8,  10,  0.4,  10],
                       controlLowerBound: List[float]=[-4],
                       controlUpperBound: List[float]=[ 4],
                       stateWeight: List[float]=[0, 0, 10, 4],
                       controlWeight: List[float]=[1],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "CartPole",
                         **kwargs)
        # Parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length

    def dynamics(self, t, x, control: List[float], action: int):

        _, x_dot, theta, theta_dot = x
        force = control[0]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        dxdt = np.zeros(len(x))
        dxdt[0] = x_dot
        dxdt[1] = xacc
        dxdt[2] = theta_dot
        dxdt[3] = thetaacc
        return dxdt

    def _get_reward(self):
        if self.truncated:
            return 0.0
        return 1.0


"""X=[x,y,dth], U=[\tau]"""
class Pendulum(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[-0.05, 0.5, -0.05],
                       initUpperBound: List[float]=[ 0.05, 0.7,  0.05],
                       goalLowerBound: List[float]=[-1.0, 0.95, -0.2],
                       goalUpperBound: List[float]=[ 1.0, 1.0,  0.2],
                       safeLowerBound: List[float]=[-1.0, -1.0, -1],
                       safeUpperBound: List[float]=[ 1.0,  1.0,  1],
                       controlLowerBound: List[float]=[-2.0],
                       controlUpperBound: List[float]= [2.0],
                       stateWeight: List[float]=[1, 1, 1],
                       controlWeight: List[float]=[1],
                       dt: float=0.05,
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Pendulum",
                         **kwargs)
        self.g = 9.81
        self.m = 1.0
        self.l = 1.0
        # self.max_speed = self.safeUpperBound[2]

    def dynamics(self, t, x, control: List[float], action: int):

        th, thdot = x  # th := theta
        u = control[0]
        self.u = u

        g = self.g
        m = self.m
        l = self.l

        newthdot = thdot + (3*g/(2*l) * np.sin(th) + 3.0/(m*l^2)*u)
        newth = th + thdot

        dxdt = np.array([newth, newthdot])
        return dxdt

    def _get_obs(self):
        state = copy.deepcopy(self.state)
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def reset(self, seed=None, options=None) -> List[float]:
        super().reset(seed=seed)
        state = self.randX0()
        while self.inTerminal(state) or self.outOfBound(state):
            state = self.randX0()
        assert(all(self.initLowerBound <= state) and all(state <= self.initUpperBound))
        theta = np.atan2(state[1], state[0])
        self.state = np.array([theta, state[2]])
        return self._get_obs()

    def _get_reward(self):
        if self.truncated:
            return 0
        x, y, thetadot = self._get_obs()
        return y


class Acrobot(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[],
                       initUpperBound: List[float]=[],
                       goalLowerBound: List[float]=[],
                       goalUpperBound: List[float]=[],
                       safeLowerBound: List[float]=[-1.0, -1.0, -1.0, -1.0, -12.5, -28],
                       safeUpperBound: List[float]=[ 1.0,  1.0,  1.0,  1.0,  12.5,  28],
                       controlLowerBound: List[float]=[-1],
                       controlUpperBound: List[float]=[ 1],
                       stateWeight: List[float]=[1, 1, 1, 1, 1, 1],
                       controlWeight: List[float]=[1],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Acrobot",
                         **kwargs)

    def dynamics(self, control: List[float], action: int, dt: float):
        raise NotImplementedError()


"""X=[x,z,θ,dx,dz,dθ], U=[f1,f2]"""
class Drone2D(OMPLEnv):
    """Taken from Honkai Dai's Paper / GitHub
    https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[ 0.2,  0.2,   np.pi*0.1, -0.05, -0.05, -0.05],
                       initUpperBound: List[float]=[ 0.25,  0.25, np.pi*0.2,  0.05,  0.05,  0.05],
                       goalLowerBound: List[float]=[-0.05, -0.05, -0.05, -0.1, -0.1, -0.1],
                       goalUpperBound: List[float]=[-0.05,  0.05,  0.05,  0.1,  0.1,  0.1],
                       safeLowerBound: List[float]=[-0.3, -0.3, -np.pi*0.3, -1.5, -1.5, -0.9],
                       safeUpperBound: List[float]=[ 0.3,  0.3,  np.pi*0.3,  1.5,  1.5,  0.9],
                       controlLowerBound: List[float]=[0, 0],
                       controlUpperBound: List[float]=[8, 8],
                       stateWeight: List[float]=[10, 5, 0, 1, 1, 0],
                       controlWeight: List[float]=[0.2, 0.2],
                       dt: float=0.1,
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Drone2D",
                         **kwargs)
        # length of the rotor arm.
        self.length = 0.25
        # mass of the quadrotor.
        self.mass = 0.486
        # moment of inertia
        self.inertia = 0.00383
        # gravity.
        self.gravity = 9.81
        self.x_equ = np.zeros(6)
        self.u_equ = np.ones(2) * (self.mass * self.gravity) / 2
        self.lqr_Q = np.diag(np.array([10, 10, 10, 1, 1, self.length / 2. / np.pi]))
        self.lqr_R = np.array([[0.1, 0.05], [0.05, 0.1]])

    def dynamics(self, t, x, control: List[float], action: int):
        self.u = control
        u = control
        q = x[:3]
        qdot = x[3:]

        qddot = np.array([
            -np.sin(q[2]) / self.mass * (u[0] + u[1]),
            np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
            self.length / self.inertia * (u[0] - u[1])
        ])
        dydt = np.hstack((qdot, qddot))
        return dydt

    def _get_reward(self):
        # if self.truncated:
        #     return 0
        act_delta = self.u - self.u_equ
        obs_delta = self.state - self.x_equ
        reward = -(act_delta).dot(self.lqr_R @ act_delta).item() - \
            obs_delta.dot(self.lqr_Q @ obs_delta).item()
        return reward


"""X=[x,y,z,roll,pitch,yaw,dx,dy,dz,droll,dpitch,dyaw], U=[f1,f2,f3,f4]"""
class Drone3D(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[],
                       initUpperBound: List[float]=[],
                       goalLowerBound: List[float]=[],
                       goalUpperBound: List[float]=[],
                       safeLowerBound: List[float]=[],
                       safeUpperBound: List[float]=[],
                       controlLowerBound: List[float]=[],
                       controlUpperBound: List[float]=[],
                       stateWeight: List[float]=[1]*12,
                       controlWeight: List[float]=[1]*4,
                       dt: float=0.1,
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "Drone3D",
                         **kwargs)
        self.mass = 0.468
        self.gravity = 9.81
        self.arm_length = 0.225
        # The inertia matrix is diagonal, we only store Ixx, Iyy and Izz.
        self.inertia = np.array([4.9E-3, 4.9E-3, 8.8E-3])
        # The ratio between the torque along the z axis versus the force.
        self.z_torque_to_force_factor = 1.1 / 29
        self.hover_thrust = self.mass * self.gravity / 4

    def dynamics(self, t, x, control: List[float], action: int):
        u = control
        rpy = x[3:6]
        pos_dot = x[6:9]
        omega = x[9:12]

        # plant_input is [total_thrust, torque_x, torque_y, torque_z] ... Eq(2)
        plant_input = np.array([[1, 1, 1, 1],
                                [0, self.arm_length, 0, -self.arm_length],
                                [-self.arm_length, 0, self.arm_length, 0],
                                [
                                    self.z_torque_to_force_factor,
                                    -self.z_torque_to_force_factor,
                                    self.z_torque_to_force_factor,
                                    -self.z_torque_to_force_factor
                                ]]) @ u
        R = rpy2rotmat(rpy)
        # m*ddx = −m*g*zW + u(0)*zB where
        # zW: z in the World Frame, i.e., [0,0,1]
        # zB: z in the Body Frame, i.e., zB=R*zW where R is a Rotation Matrix
        # Thus, ddx = [0,0,-g] + R*[0,0,u(0)]/m
        pos_ddot = np.array([
            0, 0, -self.gravity
        ]) + R @ np.array([0, 0, plant_input[0]]) / self.mass

        # Here we exploit the fact that the inertia matrix is diagonal.
        # dw = inv(Inertia)*[-(wB)xI(wB) +  u(1:3)]
        omega_dot = (np.cross(-omega, self.inertia * omega) +
                    plant_input[1:]) / self.inertia

        # Convert the angular velocity to the roll-pitch-yaw time
        # derivative.
        sin_roll = np.sin(rpy[0])
        cos_roll = np.cos(rpy[0])
        tan_pitch = np.tan(rpy[1])
        cos_pitch = np.cos(rpy[1])

        # Get more accurate rpy_dot rather than e.g., roll += droll * dt
        # Equation 2.7 in quadrotor control: modeling, nonlinear control
        # design and simulation by Francesco Sabatino
        rpy_dot = np.array(
            [[1., sin_roll * tan_pitch, cos_roll * tan_pitch],
            [0., cos_roll, -sin_roll],
            [0, sin_roll / cos_pitch, cos_roll / cos_pitch]]) @ omega

        dxdt = np.hstack((pos_dot, rpy_dot, pos_ddot, omega_dot))
        return dxdt


class CaltechDuctedFan(OMPLEnv):
    def __init__(self, pathToExecutable: str,
                       initLowerBound: List[float]=[],
                       initUpperBound: List[float]=[],
                       goalLowerBound: List[float]=[],
                       goalUpperBound: List[float]=[],
                       safeLowerBound: List[float]=[],
                       safeUpperBound: List[float]=[],
                       controlLowerBound: List[float]=[],
                       controlUpperBound: List[float]=[],
                       stateWeight: List[float]=[],
                       controlWeight: List[float]=[],
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
                         stateWeight,
                         controlWeight,
                         dt,
                         "CaltechDuctedFan",
                         **kwargs)

    def dynamics(self, control: List[float], action: int, dt: float):
        raise NotImplementedError()


register(id="ompl/Car2D-v0",
         entry_point="sciab.env.omplenv:Car2D",
         max_episode_steps=30)
register(id="ompl/DubinsCar-v0",
         entry_point="sciab.env.omplenv:DubinsCar",
         max_episode_steps=30)
register(id="ompl/DubinsCarWithAcceleration-v0",
         entry_point="sciab.env.omplenv:DubinsCarWithAcceleration",
         max_episode_steps=30)
register(id="ompl/Unicycle-v0",
         entry_point="sciab.env.omplenv:Unicycle",
         max_episode_steps=30)
register(id="ompl/UnicycleWithConstraint-v0",
         entry_point="sciab.env.omplenv:UnicycleWithConstraint",
         max_episode_steps=30)
register(id="ompl/CartPole-v0",
         entry_point="sciab.env.omplenv:CartPole",
         max_episode_steps=100)
register(id="ompl/Pendulum-v0",
         entry_point="sciab.env.omplenv:Pendulum",
         max_episode_steps=100)
register(id="ompl/Acrobot-v0",
         entry_point="sciab.env.omplenv:Acrobot",
         max_episode_steps=30)
register(id="ompl/Drone2D-v0",
         entry_point="sciab.env.omplenv:Drone2D",
         max_episode_steps=100)
register(id="ompl/Drone3D-v0",
         entry_point="sciab.env.omplenv:Drone3D",
         max_episode_steps=100)
register(id="ompl/CaltechDuctedFan-v0",
         entry_point="sciab.env.omplenv:CaltechDuctedFan",
         max_episode_steps=30)
