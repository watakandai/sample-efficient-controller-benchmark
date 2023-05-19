import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
import time
import logging
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from sciab import util


ENVNAMES = [
    "ompl/Car2D-v0",
    "ompl/DubinsCar-v0",
    "ompl/DubinsCarWithAcceleration-v0",
    "ompl/Unicycle-v0",
    "ompl/UnicycleWithConstraint-v0",
    # "ompl/CartPole-v0",
    # "ompl/Pendulum-v0",
    # "ompl/Acrobot-v0",
    # "ompl/Drone2D-v0",
    # "ompl/Drone3D-v0",
    # "ompl/CaltechDuctedFan-v0"
]


def randX0(env):
    numState = len(env.safeLowerBound)
    return env.safeLowerBound + env.np_random.random(numState) * (env.safeUpperBound - env.safeLowerBound)


def runRLExperiment(env, numSample, filedir, timeout=None):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    scale = max_action * np.ones(action_dim)
    controller = RLController(state_dim,
                              action_dim,
                              scale)
    controller.load(MODEL_FILE_DIR)

    counter = 0
    start = time.time()

    if timeout is not None and time.time() - start > timeout:
        return counter, "N/A"

    Xs = []
    successes = 0
    for i in range(numSample):

        x = randX0(env)
        X = [x]
        d = False
        counter = 0
        while not d:
            u = controller.action(x) # In our case, it will either return an int or List[float]
            nx, r, done, truncated, info = env.step(u)
            d = done or truncated
            X.append(nx)
            x = nx
            counter += 1
        Xs.append(X)
        successes += int(info["status"] == SimStatus.SIM_TERMINATED)

    if filedir:
        file = os.path.join(filedir, "trajectories", env.system, "RL")
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        np.save(file, np.array(Xs, dtype=object), allow_pickle=True)

    successRate = successes / numSample

    return successRate, time.time() - start


"""Test if OMPL oracle (Env) is Probabilistically Correct"""
def runOMPLExperiment(env, numSample, filedir, timeout=None):

    counter = 0
    start = time.time()

    if timeout is not None and time.time() - start > timeout:
        return counter, "N/A"

    trajectories = []
    for i in range(numSample):
        s = time.time()

        x = randX0(env)
        t = env.sampleBestCostTrajectory(x)
        trajectories.append(t)

        # if i % 1 == 0:
        #     print(i, x, time.time() - s)
        #     print(t)

    terminatedTrajs = list(filter(lambda t: len(t[0])>0, trajectories))
    Xs = list(map(lambda t: t[0], terminatedTrajs))
    Us = list(map(lambda t: t[2], terminatedTrajs))
    inputMean = np.mean(list(map(lambda U: np.mean(np.absolute(U), axis=0), Us)), axis=0)

    if filedir:
        file = os.path.join(filedir, "trajectories", env.system, "Voronoi")
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        np.save(file, np.array(Xs, dtype=object), allow_pickle=True)

    successRate = len(terminatedTrajs) / len(trajectories)
    print(f"Env: {env.system}, Input Mean: {inputMean}")

    return successRate, time.time() - start


def selectModulesAndRunExperiment(pathToExecutable, algorithm, envName, numSample, filedir):

    print("pathToExecutable: ", pathToExecutable, "Env: ", envName)
    env = gym.make(envName, pathToExecutable=pathToExecutable)

    msg = f"Env: {envName}, Algorithm: {algorithm}, NumSample: {numSample}"
    logging.info(msg)
    print(msg)

    if "RL" in algorithm:
        successRate, elapsedTime = runRLExperiment(env, numSample, filedir)
    else:
        successRate, elapsedTime = runOMPLExperiment(env, numSample, filedir)

    stats = [algorithm, envName, successRate, elapsedTime]
    msg = ','.join([str(col) for col in stats])
    logging.info(msg)
    print(msg)

    return stats


if __name__ == '__main__':

    args = util.parse_benchmark_args()
    algorithm = "randomVoronoi"

    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M:%S.%f')
    filedir = os.path.join("output", f"{datestr}")
    Path(filedir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(filedir, "testOracle.log"),
                        level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s')

    csv_filename = "testOracle.csv"
    column_labels = ['algorithm', 'env', 'success rate', 'time']
    util.write_header(column_labels, csv_filename, print_outputs=True)
    f = open(csv_filename, 'a+')

    for envName in ENVNAMES:
        stats = selectModulesAndRunExperiment(args.pathToExecutable,
                                              algorithm,
                                              envName,
                                              args.numSample,
                                              filedir)

        msg = ','.join([str(col) for col in stats])
        f.write(msg + '\n')
    f.close()
