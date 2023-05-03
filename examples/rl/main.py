import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
import time
import gymnasium as gym
from sciab.countersampler.base import CounterExample, FirstXOfRandomTrajSampler
from sciab.trainer.rltrainer import RLTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier
from sciab import util


def main(pathToExecutable, envName, numSample):
    print("pathToExecutable: ", pathToExecutable, "Env: ", envName)
    env = gym.make(envName, pathToExecutable=pathToExecutable)

    trainer = RLTrainer(env)
    verifier = ProbabilisticVerifier(env, numSample)
    countersampler = FirstXOfRandomTrajSampler()

    counterexample = CounterExample(x=env.reset())
    counter = 0
    start = time.time()

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

    elapsedTime = time.time() - start
    print(f"{envName}: #Samples={counter}, ElapsedTime={elapsedTime}")


if __name__ == '__main__':
    args = util.parse_args()
    main(args.pathToExecutable, args.envName, args.numSample)
