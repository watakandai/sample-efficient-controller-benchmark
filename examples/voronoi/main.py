import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
import gymnasium as gym
import argparse
from sciab.countersampler.base import BaseCounterExample, FirstXOfRandomTrajSampler
from sciab.trainer.voronoitrainer import VoronoiTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str)
    parser.add_argument("envName", type=str)
    parser.add_argument('-ns', '--numSample', type=int, default=10)
    # parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def main(pathToExecutable, envName, numSample):

    print("pathToExecutable: ", pathToExecutable, "Env: ", envName)
    env = gym.make(envName, pathToExecutable=pathToExecutable)

    trainer = VoronoiTrainer(env)
    verifier = ProbabilisticVerifier(env, numSample)
    countersampler = FirstXOfRandomTrajSampler()

    counterexample = BaseCounterExample(x=env.reset())
    counter = 0

    while True:

        print("counter: ", counter)

        print("training...")
        controller = trainer.train(counterexample)

        print("verifying...")
        result = verifier.verify(controller)

        if result.verified:
            break

        print("finding a counterexample...")
        counterexample = countersampler.sample(result)
        counter += 1


if __name__ == '__main__':
    args = parse_args()
    main(args.pathToExecutable, args.envName, args.numSample)
