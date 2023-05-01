import argparse
from sciab.env.rlenv import RLEnv, RLEnvArguments
from sciab.countersampler.base import BaseCounterExample, FirstXOfRandomTrajSampler
from sciab.trainer.rltrainer import RLTrainer
from sciab.verifier.probabilisticverifier import ProbabilisticVerifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, required=True)
    parser.add_argument('-ns', '--numSample', type=int, default=100)
    # parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def main(system, numSample):
    # TODO: OMPL Env
    eargs = RLEnvArguments(system)
    env = RLEnv(eargs)
    trainer = RLTrainer(env)
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

    print(f"{args.system}: #Samples={counter}")


if __name__ == '__main__':
    args = parse_args()
    main(args.system, args.numSample)
