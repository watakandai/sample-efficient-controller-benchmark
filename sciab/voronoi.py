import argparse
from env.omplenv import OMPLEnv, OMPLEnvArguments
from countersampler.base import BaseCounterExample, FirstXOfRandomTrajSampler
from trainer.voronoitrainer import VoronoiTrainer
from verifier.probabilisticverifier import ProbabilisticVerifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str, required=True)
    parser.add_argument("system", type=str, required=True)
    parser.add_argument('-ns', '--numSample', type=int, default=100)
    # parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def main(pathToExecutable, system, numSample):
    # TODO: OMPL Env
    eargs = OMPLEnvArguments(system, pathToExecutable)
    env = OMPLEnv(eargs)
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

    # print(f"{args.system}: No.Samples={counter}")


if __name__ == '__main__':
    args = parse_args()
    main(args.pathToExecutable, args.system, args.numSample)
