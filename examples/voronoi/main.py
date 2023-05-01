import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
import gymnasium as gym
import argparse
from sciab.env.omplenv import OMPLEnv, OMPLEnvWrapper, OMPLEnvArguments
from sciab.countersampler.base import BaseCounterExample, FirstXOfRandomTrajSampler
from sciab.trainer.voronoitrainer import VoronoiTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str)
    parser.add_argument("system", type=str)
    parser.add_argument('-ns', '--numSample', type=int, default=100)
    # parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def main(pathToExecutable, system, numSample):
    # TODO: OMPL Env
    print("pathToExecutable: ", pathToExecutable, "system: ", system)
    env = gym.make(system)
    eargs = OMPLEnvArguments(system, pathToExecutable)
    env = OMPLEnvWrapper(env, eargs)
    # env = OMPLEnv(eargs)
    trainer = VoronoiTrainer(env)
    verifier = ProbabilisticVerifier(env, numSample)
    countersampler = FirstXOfRandomTrajSampler()

    counterexample = BaseCounterExample(x=env.reset())
    counter = 0

    while True:
        controller = trainer.train(counterexample)
        result = verifier.verify(controller)
        if result.verified:
            break
        counterexample = countersampler.sample(result)
        counter += 1


if __name__ == '__main__':
    args = parse_args()
    main(args.pathToExecutable, args.system, args.numSample)
