import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToExecutable", type=str)
    parser.add_argument("envName", type=str)
    parser.add_argument('-ns', '--numSample', type=int, default=1000)
    return parser.parse_args()

