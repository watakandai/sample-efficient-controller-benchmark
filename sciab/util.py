import os
import time
import logging
import argparse
import gymnasium as gym

from sciab.countersampler.base import (
    FirstXOfRandomTrajSampler,
    RandomXOfRandomTrajSampler,
    RandomXTrajSampler,
)
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier
from sciab.countersampler.base import CounterExample
from sciab.trainer.rltrainer import RLTrainer
from sciab.trainer.voronoitrainer import VoronoiTrainer
from sciab import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("envName", type=str)
    parser.add_argument('-ns', '--numSample', type=int, default=100)
    parser.add_argument('--plot', action='store_true')

    return parser.parse_args()


def parse_benchmark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str)
    parser.add_argument('-ns', '--numSample', type=int, default=100)
    return parser.parse_args()


def selectAlgorithm(env, algorithm, filedir:str=None):
    if algorithm == "RL":
        return RLTrainer(env)
    elif "Voronoi" in algorithm:
        return VoronoiTrainer(env, filedir)
    else:
        raise Exception("Unknown algorithm: ", algorithm)


def selectCounterExampleSampler(env, algorithm):
    if algorithm == "RL":
        return FirstXOfRandomTrajSampler()
    elif "ce" in algorithm:
        return FirstXOfRandomTrajSampler()
        # return RandomXOfRandomTrajSampler()
    elif "random" in algorithm:
        return RandomXTrajSampler(env)
    else:
        raise Exception("Unknown algorithm: ", algorithm)


def write_header(csv_filename, print_outputs=True):

    column_labels = ['algorithm', 'env', '# iterations', 'time',
                     '# states', '# actions',
                     'initLB', 'initUB',
                     'termLB', 'termUB',
                     'safeLB', 'safeUB',
                     'controlLB', 'controlUB',
                     'dt']
    msg = ','.join(column_labels) + '\n'

    if os.path.exists(csv_filename):
        f = open(csv_filename, 'r')
        l = f.readline()
        f.close()
        if l == msg: return

    f = open(csv_filename, 'a+')
    f.write(msg)
    if print_outputs:
        msg = ', '.join(column_labels)
        print(msg)
        logging.debug(msg)
    f.close()


def runExperiment(env, trainer, verifier, countersampler, timeout=None):

    counter = 0
    counterexample = CounterExample(x=env.reset())
    start = time.time()

    while True:
        msg = f"counter: {counter}"
        print(msg); logging.debug(msg)

        if timeout is not None and time.time() - start > timeout:
            return counter, "N/A"

        msg = f"training..."
        logging.debug(msg)
        controller = trainer.train(counterexample)

        if counter >= trainer.startTrainingEpisode:
            msg = f"verifying..."
            logging.debug(msg)
            result = verifier.verify(controller)

            if result.verified: break

            msg = f"finding a counterexample..."
            logging.debug(msg)
            counterexample = countersampler.sample(result)
        else:
            msg = f"skipping verification..."
            logging.debug(msg)
            counterexample = CounterExample(x=env.reset())

        counter += 1

    return counter, time.time() - start


def selectModulesAndRunExperiment(pathToExecutable, algorithm, envName, numSample, filedir):

    print("pathToExecutable: ", pathToExecutable, "Env: ", envName)
    env = gym.make(envName, pathToExecutable=pathToExecutable)
    trainer = selectAlgorithm(env, algorithm, filedir=filedir)
    verifier = ProbabilisticVerifier(env, numSample, filedir=filedir)
    countersampler = selectCounterExampleSampler(env, algorithm)

    msg = f"Env: {envName}, Algorithm: {algorithm}, Verifier: {type(verifier)}, CounterSampler: {type(countersampler)} NumSample: {numSample}"
    logging.info(msg)
    print(msg)

    counter, elapsedTime = runExperiment(env, trainer, verifier, countersampler)

    stats = [algorithm, envName, counter, elapsedTime,
              env.observation_space.shape[0], env.action_space.shape[0],
              env.initLowerBound, env.initUpperBound,
              env.goalLowerBound, env.goalUpperBound,
              env.safeLowerBound, env.safeUpperBound,
              env.controlLowerBound, env.controlUpperBound,
              env.dt]

    msg = ','.join([str(col) for col in stats])
    logging.info(msg)
    print(msg)

    return stats


