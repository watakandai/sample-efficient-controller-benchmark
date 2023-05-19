import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)

import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sciab import util
from sciab.controller.rlcontroller import RLController


def randX0(env):
    numState = len(env.safeLowerBound)
    return env.safeLowerBound + env.np_random.random(numState) * (env.safeUpperBound - env.safeLowerBound)


def getLimits(Xs):
    xmin = min(map(lambda X: np.min(X, axis=0)[0], Xs))
    xmax = max(map(lambda X: np.max(X, axis=0)[0], Xs))
    ymin = min(map(lambda X: np.min(X, axis=0)[1], Xs))
    ymax = max(map(lambda X: np.max(X, axis=0)[1], Xs))
    return xmin, xmax, ymin, ymax


def getEnvLimits(env):
    xmin = env.safeLowerBound[0]
    ymin = env.safeLowerBound[1]
    xmax = env.safeUpperBound[0]
    ymax = env.safeUpperBound[1]
    return xmin, xmax, ymin, ymax


def plotTrajectories(Xs, env, filepath):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xmin, xmax, ymin, ymax = getEnvLimits(env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for X in Xs:
        ax.plot(X[:, 0], X[:, 1])
    fig.savefig(filepath)
    plt.close()


def plotFirstTransitions(Xs, env, filepath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xmin, xmax, ymin, ymax = getEnvLimits(env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for X in Xs:
        # plot only first two points of the trajectory in 2D (0,1)
        # print(X[:2, 0])
        # print(X[:2, 1])
        ax.plot(X[:2, 0], X[:2, 1])
        ax.plot(X[0, 0], X[0, 1], marker="*")
    fig.savefig(filepath)
    plt.close()


def plotVoronoi(Xs, env, filepath):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    points = list(map(lambda X: X[0, :2], Xs))
    if len(points) == 0:
        return
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax)

    xmin, xmax, ymin, ymax = getEnvLimits(env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(filepath)
    plt.close()

def runRLExperiment(env, numSample, modelFileDir):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    scale = max_action * np.ones(action_dim)
    controller = RLController(state_dim,
                              action_dim,
                              scale)
    controller.load(modelFileDir)

    Xs = []
    Us = []
    numSuccess = 0
    for i in range(numSample):

        env.reset()
        x = randX0(env)
        X = [x]
        U = []
        d = False
        while not d:
            u = controller.action(x) # In our case, it will either return an int or List[float]
            nx, r, done, truncated, info = env.step(u)
            # d = done or truncated
            d = done #or truncated
            X.append(nx)
            U.append(u)
            x = nx
        numSuccess += int(done)
        Xs.append(np.array(X))
        Us.append(np.array(U))

    successRate = numSuccess / numSample
    inputMean = np.mean(list(map(lambda U: np.mean(np.absolute(U), axis=0), Us)), axis=0)
    print(f"Env: {env.system}, successRate: {successRate}, Input Mean: {inputMean}")
    return Xs

# Env: DubinsCar, successRate: 1.0, Mean: [0.38691995]
# Env: DubinsCarWithAcceleration, successRate: 1.0, Mean: [0.70344824 0.22804886]
# Env: Car2D, successRate: 1.0, Mean: [0.6621944]
# Env: Unicycle, successRate: 1.0, Mean: [0.69036293 0.37582266]

if __name__ == '__main__':
    # filedir = Path("output/2023-05-18-10:43:15.741838")
    # filedir = Path("output/2023-05-18-22:47")
    filedir = Path("output/2023-05-19-09:52.039638")
    fd1 = filedir.joinpath("trajectories")
    fd2 = filedir.joinpath("RL")
    algorithm = "Voronoi" if fd1.exists() else "RL" if fd2.exists() else sys.exist()

    if algorithm == "Voronoi":

        filepaths = list(filter(lambda d: d.is_file() and d.suffix=='.npy', list(fd1.iterdir())))
        for fpath in filepaths:
            system = fpath.stem
            # system = "Unicycle"
            envName = f"ompl/{system}-v0"
            env = gym.make(envName, pathToExecutable="")

            filename = os.path.join(fd1, f"{system}.npy")
            Xs = np.load(filename, allow_pickle=True)

            trajFilePath = os.path.join(fd1, f"Voronoi{system}trajectories.png")
            voronoiFilePath = os.path.join(fd1,f"Voronoi{system}voronoi.png")
            transitionFilePath = os.path.join(fd1, f"Voronoi{system}firstTransitions.png")

            plotVoronoi(Xs, env, voronoiFilePath)
            plotFirstTransitions(Xs, env, transitionFilePath)
            plotTrajectories(Xs, env, trajFilePath)

    else:

        numSample = 100
        for fpath in filedir.joinpath("RL").iterdir():
            if fpath.is_file():
                continue
            system = fpath.stem
            envName = f"ompl/{system}-v0"
            env = gym.make(envName, pathToExecutable="")
            Xs = runRLExperiment(env, numSample, fpath)

            trajFilePath = os.path.join(fd2, f"RL{system}trajectories.png")
            voronoiFilePath = os.path.join(fd2, f"RL{system}voronoi.png")
            transitionFilePath = os.path.join(fd2, f"RL{system}firstTransitions.png")

            plotVoronoi(Xs, env, voronoiFilePath)
            plotFirstTransitions(Xs, env, transitionFilePath)
            plotTrajectories(Xs, env, trajFilePath)
