import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)

import time
import itertools
import gymnasium as gym
from sciab.countersampler.base import CounterExample, FirstXOfRandomTrajSampler, RandomX0TrajSampler
from sciab.trainer.voronoitrainer import RLTrainer, VoronoiTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier
from sciab import util


def run(env, trainer, verifier, countersampler, timeout=None):

    counter = 0
    counterexample = CounterExample(x=env.reset())
    start = time.time()

    while True:
        print("counter: ", counter)

        print("training...")
        controller = trainer.train(counterexample)

        print("verifying...")
        result = verifier.verify(controller)

        if result.verified:
            break

        if timeout is not None and time.time() - start > timeout:
            return counter, "N/A"

        print("finding a counterexample...")
        counterexample = countersampler.sample(result)
        counter += 1

    return counter, time.time() - start


def selectAlgorithm(env, algorithm):
    if algorithm == "RL":
        return RLTrainer(env)
    elif "Vornoi" in algorithm:
        return VoronoiTrainer(env)
    else:
        raise Exception("Unknown algorithm: ", algorithm)


def selectCounterExampleSampler(env, algorithm):
    if algorithm == "RL":
        return FirstXOfRandomTrajSampler()
    elif "ce" in algorithm:
        return FirstXOfRandomTrajSampler()
    elif "random" in algorithm:
        return RandomX0TrajSampler(env)
    else:
        raise Exception("Unknown algorithm: ", algorithm)


def write_header(csv_filename, print_outputs=True):
    f = open(csv_filename, 'a+')
    column_labels = ['algorithm', 'env', '# states', '# actions',
                     'initLB', 'initUB',
                     'termLB', 'termUB',
                     'safeLB', 'safeUB',
                     'controlLB', 'controlUB',
                     'dt', '# iterations', 'time']
    f.write(','.join(column_labels) + '\n')
    if print_outputs:
        print(', '.join(column_labels))
    f.close()


if __name__ == '__main__':

    args = util.parse_args()

    algorithms = ["RL", "ceVoronoi", "randomVoronoi"]
    envNames = ["ompl/DubinsCar-v0",
                "ompl/DubinsCarWithAcceleration-v0",
                "ompl/Unicycle-v0",
                "ompl/CartPole-v0",
                "ompl/InvertedPendulum-v0",
                "ompl/DoubleInvertedPendulum-v0",
                "ompl/Drone2D-v0",
                "ompl/Drone3D-v0",
                "ompl/CaltechDuctedFan-v0"]

    csv_filename = "benchmark.csv"
    write_header(csv_filename, print_outputs=True)
    f = open(csv_filename, 'a+')

    for (algorithm, envName) in itertools.product(algorithms, envNames):

        env = gym.make(envName, args.pathToExecutable)
        trainer = selectAlgorithm(env, algorithm)
        verifier = ProbabilisticVerifier(env, args.numSample)
        countersampler = selectCounterExampleSampler(env, algorithm)

        counter, time = run(env, trainer, verifier, countersampler)

        stats = [algorithm, envName, env.observation_space.shape[0], env.action_space.shape[0],
                 env.initLowerBound, env.initUpperBound,
                 env.termLowerBound, env.termUpperBound,
                 env.safeLowerBound, env.safeUpperBound,
                 env.controlLowerBound, env.controlUpperBound,
                 env.dt, counter, time]

        f.write(','.join([str(col) for col in stats]) + '\n')

    f.close()
