import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(CURRENT_DIR)
HOME_DIR = os.path.dirname(EXAMPLE_DIR)
sys.path.append(HOME_DIR)
import time
import logging
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from sciab.countersampler.base import CounterExample, FirstXOfRandomTrajSampler, RandomXTrajSampler
from sciab.trainer.rltrainer import RLTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier
from sciab import util


def main(pathToExecutable, envName, numSample, filedir, algorithm):
    print("pathToExecutable: ", pathToExecutable, "Env: ", envName)
    env = gym.make(envName, pathToExecutable=pathToExecutable)

    logging.info(f"Env: {envName}, NumSample: {numSample}")

    trainer = RLTrainer(env)
    verifier = ProbabilisticVerifier(env, numSample)
    # countersampler = FirstXOfRandomTrajSampler()
    countersampler = RandomXTrajSampler(env)

    counterexample = CounterExample(x=env.reset())
    counter = 0
    start = time.time()

    while True:

        msg = f"counter: {counter}"
        print(msg); logging.debug(msg)

        msg = f"training..."
        logging.debug(msg)
        controller = trainer.train(counterexample)

        msg = f"verifying..."
        logging.debug(msg)
        result = verifier.verify(controller)

        # if result.verified:
        #     break

        msg = f"finding a counterexample..."
        logging.debug(msg)
        counterexample = countersampler.sample(result)
        counter += 1

    elapsedTime = time.time() - start
    logging.info(f"{envName}: #Samples={counter}, ElapsedTime={elapsedTime}")


if __name__ == '__main__':
    args = util.parse_args()

    datestr = datetime.today().strftime('%Y-%m-%d-%H:%M')
    filedir = os.path.join("output", f"{datestr}")
    Path(filedir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(filedir, Path(__file__).parents[0].name+".log"),
                        level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s')

    main(args.pathToExecutable, args.envName, args.numSample)
